"""Grounded question answering: retrieve evidence -> prompt LLM -> resolve citations.

Citations are never produced by the model directly. The model only writes [n]
markers that refer to numbered evidence passages; file names and page numbers are
filled in from chunk metadata. Markers that do not match a passage are reported in
`invalid_citation_refs` and dropped from the citation list.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from config import get_settings
from schemas import Answer, AnswerCitation, SearchFilters, SearchResult
from search.query_expansion import has_change_intent
from services.llm_service import LLMError, OpenRouterLLM
from services.retrieval_service import RetrievalService

log = logging.getLogger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = "The information was not found in the indexed documents."

SYSTEM_PROMPT = f"""You answer questions about bid / RFP documents using ONLY the numbered evidence passages provided.

Rules:
1. Every factual sentence must end with the number(s) of the passage(s) that support it, in square
   brackets: [1], [2][3] or [1, 2]. This is the only accepted citation format: never write
   "passage 1", "(1)" or extra words inside the brackets. Cite the passage that actually contains
   the fact. When giving a list of values, cite the passage on the line that introduces the list
   or on every item.
2. Copy dates, times, time zones, amounts, model/part numbers and names exactly as written in the
   passages. Never invent or convert them, and never write file names or page numbers yourself;
   the [n] markers are resolved to sources automatically.
3. Passages are labelled with bid, file, page, document type and addendum number. Do not mix bids:
   only use passages from the bid(s) the question is about. Only state dates that appear in the
   passage text itself.
4. Addendums override the original RFP. If an addendum changes a value, give the latest value, name
   the addendum, and also state the original value with its own citation.
5. If the passages only partly answer the question, answer that part and say explicitly what is not
   stated in the documents. Do not guess or use outside knowledge. Table captions (e.g. "Table 3:
   ...") and clause numbers in a passage identify that table or clause. If the question names a
   table, clause or section that no passage labels, say so before using an unlabelled passage.
6. If the passages contain nothing relevant, reply with exactly:
   {NOT_FOUND_TOKEN}: {NOT_FOUND_MESSAGE}
7. Be concise: the direct answer first, then supporting details. Plain text, no tables.
8. Output only the final answer. Do not describe your reasoning or go through the passages one by one."""

_REF_RE = re.compile(r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# A comma is a thousands separator only before exactly three digits ("1,249.00"); otherwise
# it separates values ("pages 2,3,9" -> 2, 3, 9 - never 239).
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|\d+(?:[.:]\d+)*")
_LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}[.)]\s", re.M)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s")
# Short heading or label line, marked as such: "**Property**", "### 2010", "Investment:"
# (an unmarked short sentence fragment is not a heading)
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+[^\n]{1,60}|[*_]{2}[^*_\n]{1,60}[*_]{2}:?|[*_]{0,2}[^.!?\[\]\n:]{1,60}:[*_]{0,2})\s*$"
)
# "Label: value" line: "**Property:** 345 (2010), ..." / "Property - 2010: 345"
_KEY_VALUE_RE = re.compile(r"^\s*[*_]{0,2}[A-Za-z][^:\n]{0,60}:[*_]{0,2}\s+\S")
# A paragraph that is nothing but citation markers, optionally labelled: "[1]", "Source: [1]".
_CITATION_ONLY_RE = re.compile(r"^[\s(]*(?:\*{0,2}(?:sources?|citations?|references?|see)\*{0,2}\s*:?)?[\s.,;)]*$", re.I)
# Contextual labels: "Table 5", "Figure 2.1", "Clause 7.78", "Passage 1", "Page 2", "pp. 3-4".
_LABEL_KINDS = (r"table|figure|fig\.?|exhibit|chart|schedule|appendix|annex|attachment|section|clause"
                r"|article|item|part|note|footnote|addendum|amendment|passage|page|pages|pp?\.")
_LABEL_RE = re.compile(rf"\b({_LABEL_KINDS})\s*(?:no\.?\s*|#\s*)?(\d+(?:\.\d+)*)", re.I)
_EVIDENCE_REF_KINDS = {"passage"}  # "passage N" names an evidence passage, never a fact


def _label_kind(kind: str) -> str:
    k = kind.lower().rstrip(".")
    return {"fig": "figure", "pages": "page", "pp": "page", "p": "page"}.get(k, k)


def _ref_numbers(text: str) -> list[int]:
    """All passage numbers referenced in text: [1], [1][2], [1, 3], [2-4]."""
    out: list[int] = []
    for group in _REF_RE.findall(text):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
                out.extend(range(a, b + 1) if 0 < b - a < 20 else [a, b])
            else:
                out.append(int(part))
    return list(dict.fromkeys(out))


def _numbers(text: str) -> set[str]:
    """Numeric values in text, normalised (07 == 7, 2,000 == 2000, 2:00 kept as 2:00)."""
    out = set()
    for tok in _NUM_RE.findall(text):
        tok = tok.replace(",", "")
        if ":" in tok:
            h, _, m = tok.partition(":")
            out.add(f"{int(h)}:{m}" if h.isdigit() else tok)
        else:
            try:
                out.add(str(float(tok)).rstrip("0").rstrip(".") if "." in tok else str(int(tok)))
            except ValueError:
                out.add(tok)
    return out


def _strip_labels(text: str, question_labels: set[tuple[str, str]]) -> str:
    """Remove contextual labels whose number is not a factual claim: references to
    evidence passages ("passage 1") and labels the question itself named ("Table 5"
    when asked about Table 5). Other numbers - including a bare number that also occurs
    in the question, such as "5%" - are still checked."""
    def repl(m: re.Match) -> str:
        kind = _label_kind(m.group(1))
        if kind in _EVIDENCE_REF_KINDS or (kind, m.group(2)) in question_labels:
            return " "
        return m.group(0)
    return _LABEL_RE.sub(repl, text)


def _is_structured_line(line: str) -> bool:
    """Bullet, short heading/label ("**Property**", "### 2010"), "Label: value" line or table row."""
    return bool(_LIST_ITEM_RE.match(line) or _HEADING_LINE_RE.match(line) or _KEY_VALUE_RE.match(line)
                or " | " in line)


def _paragraphs(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into paragraphs with the passage numbers each one cites.

    Models lay out the same cited facts in different ways, so citations are associated
    with the structure they introduce or close:
      * a run of structured paragraphs (bullets, short headings, "Label: value" lines,
        table rows) inherits the citations of the cited paragraph right before it - the
        "values are as follows [1]." / "[1]:" intro - across blank lines;
      * a paragraph that is only a citation ("[1]", "Source: [1]") applies to the
        uncited paragraphs directly before it.
    Inherited citations are still checked: every value must appear in those passages.
    """
    paras = [p for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
    refs = [_ref_numbers(p) for p in paras]
    own = [bool(r) for r in refs]
    run: list[int] = []  # citations of the paragraph that introduced the current structured run
    for i, para in enumerate(paras):
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if own[i]:
            run = refs[i]
        elif lines and all(_is_structured_line(ln) for ln in lines):
            refs[i] = run
        else:
            run = []
    for i, para in enumerate(paras):
        if own[i] and _CITATION_ONLY_RE.match(_REF_RE.sub("", para)):
            j = i - 1
            while j >= 0 and not own[j] and not refs[j]:
                refs[j] = refs[i]
                j -= 1
    return list(zip(paras, refs))


def check_grounding(
    answer: str, cited: list[SearchResult], evidence: list[SearchResult], question: str = ""
) -> dict[str, list[str]]:
    """{"not_in_cited", "not_in_evidence"} - see `grounding_details` (uncited values are
    included in not_in_cited)."""
    d = grounding_details(answer, cited, evidence, question)
    return {"not_in_cited": d["not_in_cited"], "not_in_evidence": d["not_in_evidence"]}


def grounding_details(
    answer: str, cited: list[SearchResult], evidence: list[SearchResult], question: str = ""
) -> dict[str, list[str]]:
    """Find numbers/dates/times in the answer that its citations do not support.

    Citations are scoped to paragraphs: a value must appear in a passage cited in the
    same paragraph (bullet lists inherit the citations of their ":" intro line).
    Returns {"not_in_cited", "not_in_evidence", "uncited"}:
      * not_in_cited - in a cited paragraph, but not in the passages it cites (wrong
        citation if the value is elsewhere in the evidence);
      * not_in_evidence - in no retrieved passage at all (unsupported / hallucinated);
      * uncited - stated in a paragraph that cites nothing.
    `cited` is kept for API compatibility; paragraph citations are resolved against
    `evidence` (the numbered passages the model saw).
    """
    question_labels = {(_label_kind(k), n) for k, n in _LABEL_RE.findall(question or "")}

    def passage_values(r: SearchResult) -> set[str]:
        # numbers that identify the passage itself (e.g. "Addendum 1", "page 2") also count
        ident = f"{r.file_name} {r.addendum_number or ''} {r.page_number or ''} {r.page_end or ''}"
        return _numbers(r.text) | _numbers(ident)

    ev_vals = [passage_values(r) for r in evidence]
    all_vals = set().union(*ev_vals) if ev_vals else set()
    cited_ids = {r.chunk_id for r in cited}
    not_in_cited: set[str] = set()
    uncited: set[str] = set()
    values: set[str] = set()
    for para, refs in _paragraphs(answer):
        body = _LIST_MARKER_RE.sub(" ", _REF_RE.sub(" ", _strip_labels(para, question_labels)))
        vals = _numbers(body)
        values |= vals
        valid = [n for n in refs if 1 <= n <= len(evidence) and (not cited_ids or evidence[n - 1].chunk_id in cited_ids)]
        if not valid:
            uncited |= vals
            continue
        supported = set().union(*(ev_vals[n - 1] for n in valid))
        not_in_cited |= {v for v in vals if v not in supported}
    return {
        "not_in_cited": sorted(not_in_cited | uncited),
        "not_in_evidence": sorted(v for v in values if v not in all_vals),
        "uncited": sorted(uncited),
    }


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks, including an unopened one ("...reasoning</think>answer",
    when the chat template supplied the opening tag)."""
    text = _THINK_RE.sub("", text)
    if re.search(r"</think>", text, re.I):
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


# A passage is sent whole when it fits; the largest chunk is a table chunk of up to
# MAX_TABLE_CHUNK_SIZE (2400) characters plus its section heading.
EVIDENCE_PASSAGE_CHARS = 2600
# Bound on all passages of one prompt (~7.5k tokens); with many passages each one gets a
# smaller share and long ones are excerpted (see `excerpt_passage`).
EVIDENCE_TOTAL_CHARS = 30000
_MIN_PASSAGE_SHARE = 900
_STOP_FOR_FOCUS = frozenset("what which when where who whom whose were with from that this there their have does "
                            "into than then them they will shall should would could about".split())


def focus_patterns(*texts: str) -> list[re.Pattern]:
    """Word patterns (>= 4 letters, no stop words) used to pick the relevant lines of a long passage."""
    words = {w.lower() for t in texts for w in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", t or "")}
    return [re.compile(rf"\b{re.escape(w)}", re.I) for w in sorted(words - _STOP_FOR_FOCUS)]


def excerpt_passage(text: str, limit: int, patterns: list[re.Pattern] = ()) -> str:
    """`text` if it fits in `limit`; otherwise a bounded excerpt that keeps complete lines (table
    rows): the first line (section heading), then the lines matching the most `patterns`, then
    other lines in document order while space remains. Skipped text is marked with "…" and the
    kept lines stay in their original order. Never a blind head cut: relevant rows after the
    limit survive."""
    if len(text) <= limit:
        return text
    lines = text.split("\n")
    score = [sum(1 for rx in patterns if rx.search(ln)) for ln in lines]
    keep: set[int] = set()
    used = 0

    def take(i: int) -> None:
        nonlocal used
        if i not in keep and used + len(lines[i]) + 2 <= limit:
            keep.add(i)
            used += len(lines[i]) + 2

    if len(lines[0]) <= limit // 4:  # section heading / breadcrumb
        take(0)
    for i in sorted((i for i in range(len(lines)) if score[i]), key=lambda i: (-score[i], i)):
        take(i)
    best = max(range(len(lines)), key=lambda i: (score[i], -i))
    if score[best] and best not in keep:  # the best line alone is longer than the space left: window it
        line = lines[best]
        m = next(rx.search(line) for rx in patterns if rx.search(line))
        room = max(200, limit - used - 4)
        start = max(0, m.start() - room // 3)
        lines[best] = ("…" if start else "") + line[start:start + room] + ("…" if start + room < len(line) else "")
        keep.add(best)
        used += len(lines[best]) + 2
    for i in range(len(lines)):  # context in document order while space remains
        take(i)
    if not keep:  # one huge line and nothing relevant in it: its start
        return text[:limit] + " …"
    out, prev = [], -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    if prev != len(lines) - 1:
        out.append("…")
    return "\n".join(out)


def format_evidence(results: list[SearchResult], max_chars: int = EVIDENCE_PASSAGE_CHARS,
                    patterns: list[re.Pattern] = (), total_chars: int = EVIDENCE_TOTAL_CHARS) -> str:
    """Numbered evidence blocks with source metadata. Long passages are excerpted around the
    `patterns` (field cues / question words) instead of being cut at a fixed position."""
    limit = min(max_chars, max(_MIN_PASSAGE_SHARE, total_chars // max(len(results), 1)))
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = [f"bid: {r.bid_id}", f"file: {r.file_name}"]
        if r.page_number is not None:
            meta.append(f"page: {r.page_number}" + (f"-{r.page_end}" if r.page_end and r.page_end != r.page_number else ""))
        meta.append(f"type: {r.doc_type}" + (f" #{r.addendum_number}" if r.addendum_number else ""))
        # document_date is deliberately NOT shown: it often comes from PDF metadata
        # (creation date), and models then present it as the document's issue date.
        blocks.append(f"[{i}] ({'; '.join(meta)})\n{excerpt_passage(r.text, limit, patterns)}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    user = (
        f"Evidence passages:\n\n{format_evidence(results, patterns=focus_patterns(question))}\n\n"
        f"Question: {question}\n\nFinal answer only, citing passages as [n] or [n, m]:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def resolve_citations(text: str, results: list[SearchResult]) -> tuple[list[AnswerCitation], list[int]]:
    cited: list[AnswerCitation] = []
    invalid: list[int] = []
    for n in _ref_numbers(text):
        if 1 <= n <= len(results):
            r = results[n - 1]
            cited.append(
                AnswerCitation(
                    ref=n,
                    citation=r.citation,
                    file_name=r.file_name,
                    page_number=r.page_number,
                    page_end=r.page_end,
                    bid_id=r.bid_id,
                    doc_type=r.doc_type,
                    chunk_id=r.chunk_id,
                    text=r.text,
                )
            )
        else:
            invalid.append(n)
    return cited, invalid


class QAService:
    def __init__(self, retrieval: RetrievalService, llm: OpenRouterLLM):
        self.retrieval = retrieval
        self.llm = llm
        self.settings = get_settings()

    def gather_evidence(
        self,
        question: str,
        filters: Optional[SearchFilters] = None,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> list[SearchResult]:
        """Assemble answer context with high recall.

        For answering, a missed passage is worse than an extra one (the model can ignore
        noise but cannot use what it never sees). The cross-encoder sometimes demotes
        table rows and short addendum notices that plain hybrid ranks highly, so the
        context is the union of:
          1. the top `ask_context_chunks` results of the requested mode,
          2. the top `ask_hybrid_extra` plain-hybrid results (when re-ranking is used),
          3. for "revised / latest / after addendum" questions, the top
             `ask_addendum_extra` addendum passages (addendums override the RFP),
        de-duplicated and capped at `ask_max_context` passages.
        """
        s = self.settings
        mode = mode or s.default_search_mode
        ranked = [self.retrieval.search(question, top_k=top_k or s.ask_context_chunks, filters=filters, mode=mode)]
        if mode == "hybrid_rerank" and s.ask_hybrid_extra:
            ranked.append(self.retrieval.search(question, top_k=s.ask_hybrid_extra, filters=filters, mode="hybrid"))
        if s.ask_addendum_extra and has_change_intent(question) and not (filters and filters.doc_type):
            add_filters = (filters or SearchFilters()).model_copy(update={"doc_type": "addendum"})
            ranked.append(self.retrieval.search(question, top_k=s.ask_addendum_extra, filters=add_filters, mode=mode))
        seen: set[str] = set()
        evidence: list[SearchResult] = []
        for results in ranked:
            for r in results:
                if r.chunk_id not in seen and len(evidence) < max(s.ask_max_context, top_k or 0):
                    seen.add(r.chunk_id)
                    evidence.append(r.model_copy(update={"rank": len(evidence) + 1}))
        return evidence

    def ask(
        self,
        question: str,
        filters: Optional[SearchFilters] = None,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")
        return self.answer(question, self.gather_evidence(question, filters, top_k, mode), model)

    def answer(self, question: str, evidence: list[SearchResult], model: Optional[str] = None) -> Answer:
        """Answer from already-gathered evidence (prompt -> LLM -> citations -> grounding).
        `ask` = `gather_evidence` + `answer`; the Q&A agent calls this with the Retrieval
        agent's evidence."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")
        if not evidence:
            return Answer(question=question, answer=NOT_FOUND_MESSAGE, found=False, status="not_found")

        try:
            resp = self.llm.complete(build_messages(question, evidence), model=model)
        except LLMError as exc:
            log.warning("LLM call failed: %s", exc)
            return Answer(
                question=question,
                answer="",
                found=False,
                evidence=evidence,
                model=model or self.settings.llm_default_model,
                error=f"{type(exc).__name__}: {exc}",
                status="error",
            )

        text = strip_reasoning(resp.text)
        meta = dict(model=resp.model, latency_s=resp.latency_s, usage=resp.usage, finish_reason=resp.finish_reason)
        if resp.truncated:
            # Stopped by the token limit: whatever text exists is a fragment (often the
            # model's reasoning), never a validated answer. Citations are not resolved.
            return Answer(
                question=question, answer=text, found=False, evidence=evidence, status="incomplete",
                error=(f"Incomplete response: the model stopped at the token limit (finish_reason="
                       f"{resp.finish_reason}, max_tokens={self.settings.llm_max_tokens}). The text is not a "
                       "validated answer; raise LLM_MAX_TOKENS (2048 recommended) or ask again."),
                **meta,
            )
        if text.upper().lstrip("*# ").startswith(NOT_FOUND_TOKEN):
            return Answer(question=question, answer=NOT_FOUND_MESSAGE, found=False, evidence=evidence,
                          status="not_found", **meta)
        citations, invalid = resolve_citations(text, evidence)
        cited_results = [evidence[c.ref - 1] for c in citations]
        g = grounding_details(text, cited_results, evidence, question)
        uncited = g["uncited"]
        wrong_cite = [v for v in g["not_in_cited"] if v not in uncited and v not in g["not_in_evidence"]]
        problems = []
        if not citations:
            problems.append("The answer contains no valid citations; verify it against the evidence.")
        if invalid:
            problems.append(f"The answer cites passages that do not exist: {', '.join(map(str, invalid))}.")
        if g["not_in_evidence"]:
            problems.append(f"Values not found in any retrieved passage: {', '.join(g['not_in_evidence'])}")
        if wrong_cite:
            problems.append(f"Values not found in the cited passages: {', '.join(wrong_cite)}")
        if citations and uncited:
            problems.append(f"Values stated without a citation: {', '.join(uncited)}")
        return Answer(
            question=question,
            answer=text,
            found=True,
            citations=citations,
            evidence=evidence,
            invalid_citation_refs=invalid,
            ungrounded_values=g["not_in_cited"],
            error=" ".join(problems) or None,
            status="unverified" if problems else "verified",
            **meta,
        )
