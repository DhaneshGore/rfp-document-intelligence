"""HTML parsing with BeautifulSoup.

Bid portal pages (BidNet, eMMA, etc.) are mostly navigation chrome around a small
amount of content. The parser removes generic chrome (nav/header/footer/scripts and
elements whose id/class looks like menus, cookie banners, modals...) and then walks
the remaining DOM in order, emitting headings, tables, label/value pairs and text.
Nothing here is specific to a particular portal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from ingestion.encoding import decode_html
from ingestion.pdf_parser import render_table_rows
from schemas import Block, BlockType

_REMOVE_TAGS = [
    # <form> is intentionally kept: portal pages often wrap the whole notice in one.
    "script", "style", "noscript", "svg", "iframe", "nav", "header", "footer",
    "button", "select", "option", "input", "textarea", "template", "link", "meta",
]
_NOISE_ATTR = re.compile(
    r"(^|[-_\s])(nav|navbar|menu|menubar|cookie|consent|banner|modal|popup|footer|header"
    r"|breadcrumb|sidebar|share|social|skip|toolbar|dropdown|login|avatar|print-options|tooltip)"
    r"([-_\s]|$)",
    re.I,
)
_LABEL_ATTR = re.compile(r"(^|[-_\s])(label|field-label|key|term|name)([-_\s]|$)", re.I)
_BLOCK_TAGS = {"p", "li", "pre", "blockquote", "address", "dd", "dt", "td", "th", "caption"}
_HEADING_RE = re.compile(r"^h([1-6])$")
_UI_NOISE = {
    "loading...", "ok", "cancel", "delete", "follow", "print", "share", "mail", "email",
    "facebook", "linkedin", "twitter", "see more", "see less", "(opens in a new window)",
    "search", "help", "logout",
}
_UI_NOISE_RE = re.compile(
    r"^(are you sure you want to|we use cookies|this site uses cookies|skip to main content)", re.I
)


def _attr_text(tag: Tag) -> str:
    classes = tag.get("class") or []
    return " ".join([tag.get("id") or "", " ".join(classes), tag.get("role") or ""])


def _clean(text: str) -> str:
    return " ".join(text.split())


def _is_noise_container(tag: Tag) -> bool:
    if tag.name in ("body", "html", "main", "article"):
        return False
    if tag.get("role") in ("navigation", "banner", "contentinfo", "dialog", "menu", "menubar"):
        return True
    if tag.get("aria-hidden") == "true" or "display:none" in (tag.get("style") or "").replace(" ", ""):
        return True
    return bool(_NOISE_ATTR.search(_attr_text(tag)))


def _find_main(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "[role=main]", "article", "#content", "#main", ".content", ".main"):
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 200:
            return el
    return soup.body or soup


_JSON_SKIP_KEY = re.compile(
    r"(^|_)(id|uuid|guid|hash|key|token|nonce|class(name)?|style|css|src|href|url|uri|icon|image|img|logo"
    r"|locale|lang|type|mime|format|version|build|etag|checksum)s?$", re.I,
)


def _json_strings(value, key: str = "", out: Optional[list[str]] = None) -> list[str]:
    """Human-readable strings from embedded JSON (JSON-LD, app state): "key: value"."""
    out = [] if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            if not str(k).startswith(("@", "_")):
                _json_strings(v, str(k), out)
    elif isinstance(value, list):
        for v in value:
            _json_strings(v, key, out)
    elif isinstance(value, str):
        if _JSON_SKIP_KEY.search(key):
            return out
        text =_clean(BeautifulSoup(value, "lxml").get_text(" ")) if "<" in value else _clean(value)
        # skip ids, urls, css classes, base64...: keep values that read like text
        if (len(text.split()) >= 2 or (key and len(text) >= 4 and any(ch.isdigit() for ch in text))) \
                and not text.startswith(("http://", "https://", "/")) and len(text) <= 5000:
            label = re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", key).strip().capitalize()
            out.append(f"{label}: {text}" if label and len(text) < 200 else text)
    return out


def _scripted_page_fallback(soup: BeautifulSoup) -> list[Block]:
    """Text a JavaScript-rendered page still carries without running its scripts:
    <noscript> content, meta description / Open Graph tags and embedded JSON data."""
    blocks: list[Block] = []
    for name in ("description", "og:description", "twitter:description"):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and _clean(tag.get("content") or ""):
            blocks.append(Block(type=BlockType.TEXT, text=_clean(tag["content"])))
    for ns in soup.find_all("noscript"):
        txt = _clean(ns.get_text(" "))
        if len(txt.split()) >= 5 and not re.search(r"enable javascript|javascript (is )?(required|disabled)", txt, re.I):
            blocks.append(Block(type=BlockType.TEXT, text=txt))
    for sc in soup.find_all("script", attrs={"type": re.compile(r"json", re.I)}):
        try:
            data = json.loads(sc.string or sc.get_text() or "")
        except ValueError:
            continue
        blocks += [Block(type=BlockType.TEXT, text=t) for t in _json_strings(data)[:500]]
    return blocks


class HtmlParser:
    def parse(self, path: Path) -> tuple[list[Block], Optional[str]]:
        blocks, title, _ = self.parse_with_info(path)
        return blocks, title

    def parse_with_info(self, path: Path) -> tuple[list[Block], Optional[str], list[str]]:
        """Like parse(), plus warnings. The file's encoding is detected (BOM, UTF-16,
        <meta charset>, UTF-8, Windows-1252) instead of assuming UTF-8."""
        return self.parse_html_with_info(decode_html(Path(path).read_bytes()))

    def parse_html(self, html: str) -> tuple[list[Block], Optional[str]]:
        blocks, title, _ = self.parse_html_with_info(html)
        return blocks, title

    def parse_html_with_info(self, html: str) -> tuple[list[Block], Optional[str], list[str]]:
        soup = BeautifulSoup(html, "lxml")
        title = _clean(soup.title.get_text()) if soup.title else None
        has_scripts = bool(soup.find("script", src=True) or len(soup.find_all("script")) >= 2)
        fallback = _scripted_page_fallback(soup)  # collected before scripts/noscript are removed
        for t in soup(_REMOVE_TAGS):
            t.decompose()
        for c in list(soup.find_all(string=lambda s: s.__class__.__name__ == "Comment")):
            c.extract()
        for t in list(soup.find_all(True)):
            if t.parent is not None and _is_noise_container(t):
                t.decompose()

        blocks: list[Block] = []
        self._walk(_find_main(soup), blocks)
        # Drop UI words and consecutive duplicates.
        out: list[Block] = []
        seen_recent: list[str] = []
        for b in blocks:
            key = b.text.lower()
            if key in _UI_NOISE or len(key) < 2 or _UI_NOISE_RE.match(key):
                continue
            if key in seen_recent:
                continue
            out.append(b)
            seen_recent = (seen_recent + [key])[-5:]

        warnings: list[str] = []
        body_chars = sum(len(b.text) for b in out)
        if body_chars < 200 and (has_scripts or fallback):
            known = {b.text for b in out}
            extra = [b for b in fallback if b.text not in known]
            if extra:
                if title and title not in known:
                    out.insert(0, Block(type=BlockType.HEADING, text=title, level=1))
                out += extra
                warnings.append(
                    "Page body is mostly generated by JavaScript; indexed the text embedded in the saved file "
                    "(meta description, <noscript>, JSON data). Content loaded by scripts at view time is missing."
                )
            elif not body_chars:
                warnings.append(
                    "Page content is generated by JavaScript and is not in the saved file. Save the page from a "
                    "browser after it has loaded (Save as > Webpage, Complete) or print it to PDF, then upload that."
                )
        return out, title, warnings

    def _walk(self, node: Tag, out: list[Block]) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                txt = _clean(str(child))
                if txt:
                    out.append(Block(type=BlockType.TEXT, text=txt))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            m = _HEADING_RE.match(name)
            if m:
                txt = _clean(child.get_text(" "))
                if txt:
                    out.append(Block(type=BlockType.HEADING, text=txt, level=int(m.group(1))))
                continue
            if name == "table":
                self._table(child, out)
                continue
            if name == "dl":
                for dt in child.find_all("dt"):
                    dd = dt.find_next_sibling("dd")
                    val = _clean(dd.get_text(" ")) if dd else ""
                    out.append(Block(type=BlockType.TEXT, text=f"{_clean(dt.get_text(' '))}: {val}".strip(": ")))
                continue
            pair = self._label_value(child)
            if pair:
                out.append(Block(type=BlockType.TEXT, text=pair))
                continue
            if name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    txt = _clean(li.get_text(" "))
                    if txt:
                        out.append(Block(type=BlockType.LIST_ITEM, text=txt))
                continue
            if name in _BLOCK_TAGS or not child.find(True):
                txt = _clean(child.get_text(" "))
                if txt:
                    out.append(Block(type=BlockType.TEXT, text=txt))
                continue
            self._walk(child, out)

    @staticmethod
    def _label_value(tag: Tag) -> Optional[str]:
        """Detect `<x class="...label">Label</x><y>Value</y>` field pairs."""
        kids = [k for k in tag.children if isinstance(k, Tag)]
        if len(kids) != 2:
            return None
        label, value = kids
        if not _LABEL_ATTR.search(_attr_text(label)) and label.name not in ("label", "dt", "th"):
            return None
        ltxt, vtxt = _clean(label.get_text(" ")), _clean(value.get_text(" "))
        if not ltxt or len(ltxt) > 80:
            return None
        return f"{ltxt.rstrip(':')}: {vtxt}" if vtxt else None

    @staticmethod
    def _table(table: Tag, out: list[Block]) -> None:
        grid: list[list[str]] = []
        header_rows = 0
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            if all(c.name == "th" for c in cells) and len(grid) == header_rows:
                header_rows += 1
            grid.append([_clean(c.get_text(" ")) for c in cells])
        rows = render_table_rows(grid)
        if not rows:
            return
        header_rows = min(header_rows, len(rows) - 1) if len(rows) > 1 else 0
        header = "\n".join(rows[:header_rows]) or None
        body = rows[header_rows:]
        out.append(
            Block(
                type=BlockType.TABLE,
                text="\n".join(([header] if header else []) + body),
                table_header=header,
                table_rows=body,
            )
        )
