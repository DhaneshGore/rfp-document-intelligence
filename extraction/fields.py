"""The 20 fields of assignment Section 8 - the single place that defines them.

Each field has the assignment's exact name and description, the extraction group it is
requested in (one LLM call per group), the search queries used to collect its evidence,
whether an addendum may change it, the kind of value it holds (which selects the
validation rules in `extraction.service`), and two evidence signals used by the focused
retry stage:
  cue       regex: a passage matching it actually addresses the field (wording, not rank),
            so a `not_found` answer over such a passage deserves one focused retry;
  coverage  (label, regex) pairs: if the field's evidence matches the regex but the value
            does not, the value is incomplete (e.g. a warranty stated but not extracted).
Nothing here refers to a specific bid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Value kinds -> validation rules (see extraction.service.validate_value):
#   id       identifier(s) that must appear verbatim in a cited passage and contain a digit
#   date     must contain a parseable date; numbers grounded in the cited passages
#   contact  must contain an email or phone number that appears in a cited passage
#   text     free text; most of its words must come from the cited passages
#   summary  generated summary; numbers grounded, looser word overlap
KINDS = ("id", "date", "contact", "text", "summary")

GROUPS = {
    "identity": "Bid identity and contacts",
    "dates_logistics": "Dates, submission and logistics",
    "commercial_legal": "Commercial and legal terms",
    "product": "Products and specifications",
    "summary": "Bid summary",
}


# Payment clauses that are not the agency's payment terms: prime vendor <-> subcontractor /
# supplier duties (M/WBE programs, retainage, lien releases) and seller financing.
SUBCONTRACT_PAYMENT = (r"\b(sub-?contract\w*|sub-?consultants?|prime (vendors?|contractors?)|M/?WBE|"
                       r"retainage|lien releases?|financing agreement)\b")
# The bid submission deadline mixed into a delivery value ("scope table shows due date 06/10/2024");
# a clause that also mentions delivery ("delivery due date 07/01/2024") is not matched.
SUBMISSION_DEADLINE = (r"^(?!.*\bdeliver).*\b(due date|closing date|submission deadline|bid opening|"
                       r"(bids?|proposals?|responses?) (are |is )?due)\b")
# Documents issued by the buyer (vs. seller documents such as quotes / spec sheets, doc_type "specs").
SOLICITATION_DOCS = ("rfp", "addendum", "bid_page")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    group: str
    queries: tuple[str, ...]
    kind: str = "text"
    addendum_sensitive: bool = False  # an addendum may change it -> also search addendums
    multi: bool = False  # value may be a list (several model numbers, documents, ...)
    hint: str = ""  # extra extraction guidance shown after the assignment's description
    cue: str = ""  # regex: evidence passage that addresses this field (see module docstring)
    cue_exclude: str = ""  # regex: a cue match in a sentence matching this does not count
    reject_value: str = ""  # regex: a value matching this describes something else -> rejected
    # buyer-requirement fields: solicitation documents outrank seller documents (quotes, spec sheets)
    prefer_doc_types: tuple[str, ...] = ()
    coverage: tuple[tuple[str, str], ...] = ()  # (label, regex) completeness checks
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: unknown group {self.group!r}")


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("Bid Number", "Official solicitation / RFP identifier (e.g. JA-207652).", "identity",
              ("solicitation number bid number RFP RFQ IFB number", "request for proposal number"), kind="id",
              cue=r"\b(solicitation|bid|rfp|rfq|ifb|porfp|tender)\s*(number|no\.?|#)"),
    FieldSpec("Title", "Full title of the bid or RFP.", "identity",
              ("title of the solicitation request for proposal", "bid information title project name"),
              cue=r"\btitle\s*:"),
    FieldSpec("Due Date", "Final submission deadline, including time and time zone. Must reflect any addendum "
              "changes.", "dates_logistics",
              ("proposal due date submission deadline closing date and time", "due date extended changed"),
              kind="date", addendum_sensitive=True, cue=r"\b(due date|closing date|deadline|proposals? (are )?due)\b"),
    FieldSpec("Bid Submission Type", "How the bid is submitted (electronic portal, sealed paper, email, etc.).",
              "dates_logistics", ("how to submit proposals electronically sealed envelope email portal",),
              addendum_sensitive=True, cue=r"\b(submit\w*|submission)\b[^.]{0,80}\b(electronic\w*|online|portal|"
                                           r"sealed|e-?mail|paper|hard cop\w*)\b"),
    FieldSpec("Term of Bid", "Contract duration and any renewal options.", "dates_logistics",
              ("contract term duration renewal options years",),
              cue=r"\b(contract term|initial term|term of (the )?(contract|agreement)|renewal|length of contract)\b"),
    FieldSpec("Pre Bid Meeting", "Date, time, location and whether attendance is mandatory, or \"None\".",
              "dates_logistics", ("pre-bid meeting pre-proposal conference date time location mandatory",),
              addendum_sensitive=True, cue=r"\bpre-?\s?(bid|proposal)\b"),
    FieldSpec("Installation", "Whether installation / deployment / imaging services are required.",
              "dates_logistics", ("installation deployment imaging setup services required",),
              cue=r"\b(installation|deployment|imaging|white[- ]glove)\b"),
    FieldSpec("Bid Bond Requirement", "Required bid bond or security, amount or percentage, or \"Not required\".",
              "commercial_legal", ("bid bond bid security guarantee percent required",),
              addendum_sensitive=True, cue=r"\b(bid|proposal) (bond|security|guarant\w*)\b"),
    FieldSpec("Delivery Date", "Required delivery date or delivery window after award.", "dates_logistics",
              ("delivery days after award delivery date schedule", "delivered within days of award"),
              addendum_sensitive=True,
              hint="Only when the goods must be delivered; the bid submission due date is a different field.",
              cue=r"\bdeliver(y|ed|ies)?\b[^.|]{0,60}\b(within|days?|weeks?|by|date|after award|ship)\b",
              reject_value=SUBMISSION_DEADLINE),
    FieldSpec("Payment Terms", "Payment terms (e.g. Net 30), invoicing conditions.", "commercial_legal",
              ("payment terms net 30 invoices invoicing", "invoice submitted within days payment"),
              hint="Only the buying agency's terms for paying the vendor and how the vendor must invoice the "
                   "agency. Payment duties between a prime vendor and its subcontractors / suppliers (M/WBE, "
                   "retainage, prompt payment to subcontractors) or a seller's financing terms are NOT the "
                   "agency's payment terms; if only those are stated, return null.",
              cue=r"\b(net\s?\d+|invoic\w*|payment terms?)\b", cue_exclude=SUBCONTRACT_PAYMENT,
              reject_value=SUBCONTRACT_PAYMENT, prefer_doc_types=SOLICITATION_DOCS),
    FieldSpec("Any Additional Documentation Required", "Affidavits, forms, certificates, insurance, W-9, etc.",
              "commercial_legal",
              ("required forms certificates affidavits",
               "form certificate must be completed and submitted with the response",
               "required forms affidavits certificates insurance W-9 submitted with proposal",
               "documents attachments that must be included"), addendum_sensitive=True, multi=True,
              hint="List every form, certificate, affidavit, insurance or registration document the bidder must "
                   "submit, including ones an addendum says to submit.",
              cue=r"\b(affidavit|certificate|certification|W-?9|insurance|form \w*\d)\b"),
    FieldSpec("MFG for Registration", "Manufacturer(s) the bidder must be registered / authorized with.",
              "commercial_legal", ("manufacturer authorized reseller registered certified partner",), multi=True,
              cue=r"\b(authori[sz]ed|certified) (reseller|dealer|partner|distributor)\b|\bmanufacturer name\b"),
    FieldSpec("Contract or Cooperative to use", "Referenced cooperative / state contract vehicle, if any.",
              "commercial_legal",
              ("cooperative purchasing contract vehicle state contract master contract interlocal",),
              cue=r"\b(cooperative|master contract|state contract|interlocal|piggyback)\b"),
    FieldSpec("Model_no", "Model number(s) of the requested product(s).", "product",
              ("model number model", "product model configuration"), kind="id", addendum_sensitive=True,
              multi=True, cue=r"\bmodel\s*(#|no\.?|number)"),
    FieldSpec("Part_no", "Part / SKU number(s) of the requested product(s).", "product",
              ("part number SKU manufacturer part", "SKU list quote"), kind="id", addendum_sensitive=True,
              multi=True, cue=r"\b(SKU|part\s*(#|no\.?|number))\b"),
    FieldSpec("Product", "Product name(s) and quantity.", "product",
              ("products quantity items requested", "quantity of devices units",
               "product name description model qty quantity"), addendum_sensitive=True,
              multi=True, hint="List every line item requested - including accessories - with its quantity, model "
                               "and part numbers where stated. Use the RFP's item / pricing table together with "
                               "any quote or spec sheet, and cite every source used. Do not invent quantities.",
              cue=r"\b(qty|quantity|product name)\b"),
    FieldSpec("contact_info", "Procurement contact: name, email, phone.", "identity",
              ("buyer name email", "procurement contact buyer name email phone",
               "questions must be directed to contact"),
              kind="contact", hint="Prefer the named buyer / procurement officer; add a general procurement "
                                   "contact only in addition.",
              cue=r"\b(buyer|point of contact|procurement officer|contact information)\b"),
    FieldSpec("company_name", "Issuing organization / agency name.", "identity",
              ("issuing agency organization district name", "issued by purchasing department"),
              cue=r"\b(issuing organi[sz]ation|agency name|issued by)\b"),
    FieldSpec("Bid Summary", "A concise (3-6 sentence) summary of the bid, generated by the system.", "summary",
              ("scope of work purpose overview of the solicitation", "introduction background requirements"),
              kind="summary"),
    FieldSpec("Product Specification", "Key technical specifications (CPU, RAM, storage, display, warranty, etc.).",
              "product", ("technical specifications processor memory storage display warranty",
                          "minimum requirements processor memory storage display",
                          "minimum specifications configuration",
                          "warranty years extended warranty coverage"), addendum_sensitive=True,
              hint="One short line per product or tier; include the warranty duration and coverage where stated.",
              cue=r"\b(processor|cpu|memory|ram|storage|ssd|display)\b",
              coverage=(("warranty", r"\bwarrant(y|ies)\b"),)),
)

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)
BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}


def fields_in(group: str) -> list[FieldSpec]:
    return [f for f in FIELDS if f.group == group]


assert len(FIELDS) == 20 and len(BY_NAME) == 20, "Section 8 defines exactly 20 fields"
