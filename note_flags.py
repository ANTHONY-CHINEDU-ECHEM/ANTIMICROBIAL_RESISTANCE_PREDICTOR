"""Lightweight keyword/regex note-derived risk-flag extractor.

Stands in for a fine-tuned ClinicalBERT extraction stage: given a free-text
clinical note, flags prior-MDRO mentions, recent travel, and
immunosuppression — the same three signals the synthetic generator encodes
directly as columns. This module demonstrates the extraction contract a
real ClinicalBERT (or even a simpler regex/NER pipeline) would need to
satisfy to plug into `src/train.py` unchanged.
"""
import re

MDRO_PATTERNS = [r"\bmrsa\b", r"\bvre\b", r"\bmulti-?drug[- ]resistant\b", r"\besbl\b", r"\bcre\b"]
TRAVEL_PATTERNS = [r"\btravel(?:ed|led)?\s+to\b", r"\brecent\s+travel\b", r"\bforeign\s+hospital(?:ization)?\b"]
IMMUNOSUPPRESSED_PATTERNS = [r"\bchemotherapy\b", r"\btransplant\b", r"\bimmunosuppress\w*\b",
                              r"\bneutropeni\w*\b", r"\bhiv\b", r"\bcorticosteroid\w*\b"]


def _any_match(patterns, text):
    text = text.lower()
    return any(re.search(p, text) for p in patterns)


def extract_flags(note_text: str) -> dict:
    return {
        "note_flag_prior_mdro": int(_any_match(MDRO_PATTERNS, note_text)),
        "note_flag_recent_travel": int(_any_match(TRAVEL_PATTERNS, note_text)),
        "note_flag_immunosuppressed": int(_any_match(IMMUNOSUPPRESSED_PATTERNS, note_text)),
    }


if __name__ == "__main__":
    sample = "Patient with history of MRSA colonization, recent travel to India, on chemotherapy."
    print(extract_flags(sample))
