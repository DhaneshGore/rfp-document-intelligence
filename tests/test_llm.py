"""LLM service tests with a mocked OpenRouter (httpx.MockTransport) - no network/API key."""
import json

import httpx
import pytest

from config import Settings
from services.llm_service import (
    UNVERIFIED,
    VERIFIED_FREE,
    LLMConfigError,
    LLMInvalidResponseError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    ModelNotAllowedError,
    OpenRouterLLM,
)

FREE_A = "vendor/model-a:free"
FREE_B = "vendor/model-b:free"
PAID_FREE_SUFFIX = "vendor/sneaky:free"  # has suffix but catalogue says it costs money
PAID = "vendor/paid-model"

CATALOG = {"data": [
    {"id": FREE_A, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": FREE_B, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": PAID_FREE_SUFFIX, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    {"id": PAID, "pricing": {"prompt": "0.00001", "completion": "0.00003"}},
]}


def ok_body(model, text="Answer [1]"):
    return {"model": model, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}}


def make_llm(handler, models=(FREE_A, FREE_B), fallback=(), key="sk-test", catalog=CATALOG, **kw):
    calls = []

    def route(request: httpx.Request):
        if request.url.path.endswith("/models"):
            if catalog is None:
                raise httpx.ConnectError("offline")
            return httpx.Response(200, json=catalog)
        body = json.loads(request.content)
        calls.append(body)
        return handler(body, len(calls))

    s = Settings(openrouter_api_key=key, llm_models=list(models), llm_default_model=models[0],
                 llm_fallback_models=list(fallback), llm_max_retries=1, **kw)
    llm = OpenRouterLLM(s, http=httpx.Client(transport=httpx.MockTransport(route)), sleep=lambda s: None)
    return llm, calls


MSG = [{"role": "user", "content": "hi"}]


def test_success_tracks_usage_latency_and_caps_price():
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])))
    r = llm.complete(MSG)
    assert r.text == "Answer [1]" and r.model == FREE_A
    assert r.usage.total_tokens == 128 and r.latency_s >= 0
    assert calls[0]["provider"] == {"max_price": {"prompt": 0, "completion": 0}}
    assert len(calls) == 1  # only the selected model is called


def test_rate_limit_retries_then_succeeds():
    def h(b, n):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"Retry-After": "1"}) if n == 1 \
            else httpx.Response(200, json=ok_body(b["model"]))
    llm, calls = make_llm(h)
    assert llm.complete(MSG).text == "Answer [1]"
    assert len(calls) == 2


def test_rate_limit_exhausted_without_fallback_raises():
    llm, calls = make_llm(lambda b, n: httpx.Response(429, json={"error": {"message": "limit"}}))
    with pytest.raises(LLMRateLimitError):
        llm.complete(MSG)
    assert {c["model"] for c in calls} == {FREE_A}  # no silent switch to another model


def test_explicit_fallback_used_on_provider_error():
    def h(b, n):
        if b["model"] == FREE_A:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=ok_body(b["model"], "from B [1]"))
    llm, calls = make_llm(h, fallback=(FREE_B,))
    r = llm.complete(MSG)
    assert r.model == FREE_B and any("overloaded" in str(a.get("error")) for a in r.attempts)


def test_timeout_then_fallback():
    def h(b, n):
        if b["model"] == FREE_A:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=ok_body(b["model"]))
    llm, _ = make_llm(h, fallback=(FREE_B,))
    assert llm.complete(MSG).model == FREE_B


def test_paid_models_are_refused():
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])),
                          models=(FREE_A, PAID_FREE_SUFFIX, PAID))
    with pytest.raises(ModelNotAllowedError, match="not free"):
        llm.complete(MSG, model=PAID_FREE_SUFFIX)
    with pytest.raises(ModelNotAllowedError, match=":free"):
        llm.complete(MSG, model=PAID)
    with pytest.raises(ModelNotAllowedError, match="not in the configured"):
        llm.complete(MSG, model="other/model:free")
    assert calls == []  # nothing was ever sent


def test_paid_fallback_is_skipped_not_used():
    def h(b, n):
        return httpx.Response(503, json={"error": {"message": "down"}})
    llm, calls = make_llm(h, fallback=(PAID_FREE_SUFFIX,))
    with pytest.raises(Exception):
        llm.complete(MSG)
    assert {c["model"] for c in calls} == {FREE_A}


def test_402_payment_required_is_not_retried():
    llm, calls = make_llm(lambda b, n: httpx.Response(402, json={"error": {"message": "insufficient credits"}}),
                          fallback=(FREE_B,))
    with pytest.raises(ModelNotAllowedError):
        llm.complete(MSG)
    assert len(calls) == 1


def test_missing_api_key():
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])), key="")
    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        llm.complete(MSG)
    assert calls == []


def test_invalid_key_does_not_fall_back():
    llm, calls = make_llm(lambda b, n: httpx.Response(401, json={"error": {"message": "bad key"}}),
                          fallback=(FREE_B,))
    with pytest.raises(LLMConfigError):
        llm.complete(MSG)
    assert len(calls) == 1


def test_403_model_restriction_is_not_reported_as_bad_key():
    def h(b, n):
        if b["model"] == FREE_A:
            return httpx.Response(403, json={"error": {"message": "only available on agentic harnesses"}})
        return httpx.Response(200, json=ok_body(b["model"]))
    llm, _ = make_llm(h)
    with pytest.raises(LLMModelUnavailableError, match="agentic harnesses"):
        llm.complete(MSG)
    llm, _ = make_llm(h, fallback=(FREE_B,))
    assert llm.complete(MSG).model == FREE_B  # explicit fallback is allowed for this case


def test_403_auth_message_is_config_error():
    llm, _ = make_llm(lambda b, n: httpx.Response(403, json={"error": {"message": "Invalid API key"}}))
    with pytest.raises(LLMConfigError):
        llm.complete(MSG)


def test_reasoning_budget_exhausted_has_actionable_message():
    body = {"choices": [{"message": {"content": None, "reasoning": "thinking..."}, "finish_reason": "length"}]}
    llm, _ = make_llm(lambda b, n: httpx.Response(200, json=body))
    with pytest.raises(LLMInvalidResponseError, match="LLM_MAX_TOKENS"):
        llm.complete(MSG)


def test_payload_requests_usage_and_reasoning_effort():
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])), llm_reasoning_effort="low")
    llm.complete(MSG)
    assert calls[0]["usage"] == {"include": True}
    assert calls[0]["reasoning"] == {"effort": "low", "exclude": True}
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])), llm_reasoning_effort="")
    llm.complete(MSG)
    assert "reasoning" not in calls[0]


def test_upstream_metadata_included_in_error():
    body = {"error": {"code": 429, "message": "Provider returned error",
                      "metadata": {"provider_name": "Google AI Studio", "raw": "temporarily rate-limited upstream"}}}
    llm, _ = make_llm(lambda b, n: httpx.Response(429, json=body))
    with pytest.raises(LLMRateLimitError, match="Google AI Studio.*rate-limited upstream"):
        llm.complete(MSG)


def test_cost_is_recorded():
    body = ok_body(FREE_A)
    body["usage"]["cost"] = 0
    llm, _ = make_llm(lambda b, n: httpx.Response(200, json=body))
    assert llm.complete(MSG).usage.cost == 0


@pytest.mark.parametrize("body", [
    {"choices": []},
    {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
    {"unexpected": True},
])
def test_invalid_responses(body):
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=body))
    with pytest.raises(LLMInvalidResponseError):
        llm.complete(MSG)
    assert len(calls) == 2  # retried once (llm_max_retries=1)


def test_non_json_response():
    llm, _ = make_llm(lambda b, n: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(LLMInvalidResponseError):
        llm.complete(MSG)


def test_error_object_inside_200():
    llm, _ = make_llm(lambda b, n: httpx.Response(200, json={"error": {"code": 404, "message": "No endpoints"}}))
    with pytest.raises(LLMModelUnavailableError):
        llm.complete(MSG)


def test_unlisted_model_unavailable():
    cat = {"data": [{"id": FREE_B, "pricing": {"prompt": "0", "completion": "0"}}]}
    llm, calls = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])), catalog=cat)
    with pytest.raises(LLMModelUnavailableError):
        llm.complete(MSG)
    assert calls == []


def test_catalog_offline_marks_unverified_but_allows_free_suffix():
    llm, _ = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])), catalog=None)
    statuses = {m["id"]: m["status"] for m in llm.model_statuses()}
    assert statuses[FREE_A] == UNVERIFIED
    r = llm.complete(MSG)
    assert any(a.get("note") == "free status unverified" for a in r.attempts)


def test_model_statuses_verified():
    llm, _ = make_llm(lambda b, n: httpx.Response(200, json=ok_body(b["model"])),
                      models=(FREE_A, PAID_FREE_SUFFIX))
    st = {m["id"]: m for m in llm.model_statuses()}
    assert st[FREE_A]["status"] == VERIFIED_FREE and st[FREE_A]["usable"] and st[FREE_A]["default"]
    assert st[PAID_FREE_SUFFIX]["status"] == "not_free" and not st[PAID_FREE_SUFFIX]["usable"]


def test_default_config_has_six_free_suffixed_models():
    s = Settings(_env_file=None)
    assert len(s.llm_models) >= 6
    assert all(m.endswith(":free") for m in s.llm_models)
    assert s.llm_default_model in s.llm_models
    assert s.llm_fallback_models == []  # no implicit fallback


def test_env_list_parsing(monkeypatch):
    monkeypatch.setenv("LLM_MODELS", f"{FREE_A}, {FREE_B}")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", json.dumps([FREE_B]))
    s = Settings(_env_file=None)
    assert s.llm_models == [FREE_A, FREE_B] and s.llm_fallback_models == [FREE_B]
