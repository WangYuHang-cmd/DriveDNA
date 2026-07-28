#!/usr/bin/env python3
"""
Model consolidation: map raw carFingerprint directory names -> nameplate-level canonical models.

Merges year / generation / trim / powertrain variants of the SAME nameplate, while keeping
model-identifying names distinct (Tesla Model 3/Y/S/X separate; Hyundai Ioniq 5/6/9 separate;
Rivian R1 vs R1S separate). Reporting layer only — raw fingerprints are unchanged.

    from model_merge import canon
    canon("HONDA_CIVIC_2022")   # -> "HONDA_CIVIC"
    canon("TESLA_AP3_MODEL_3")  # -> "TESLA_MODEL_3"
"""
import re

# tokens stripped as year / generation / trim / powertrain (model-identifying names are KEPT).
# NOTE: do NOT strip `HD` (Ram HD is a distinct model) or `G\d+` (Genesis G70/G80 are models).
STRIP = set("""BOSCH ACC PEDAL CAMERA ASCM PREMIER RAVEN FL MMR NMS PE NON SCC SW
HEV PHEV HYBRID EV ELECTRIC PRIME PREGLOBAL TSS2 CHN RANGE""".split())
GENRE = re.compile(r'^(MK\d+|GEN\d*|\d+(ST|ND|RD|TH)|\d+G|T\d+)$')  # \d+G = Honda 5G/6G/11G gen
YEAR = re.compile(r'^(19|20)\d{2}$')

# irregular names that need explicit normalization
SPECIAL = {
    "KiaNiro2023": "KIA_NIRO",
    "TESLA_MODELY": "TESLA_MODEL_Y",
    "TESLA_AP3_MODEL_3": "TESLA_MODEL_3",
    "TESLA_MODEL_S_RAVEN": "TESLA_MODEL_S",
    "FORD_ESCAPE_MK4_5": "FORD_ESCAPE",       # MK4.5 facelift -> stray '5'
    "HYUNDAI_IONIQ_5_N": "HYUNDAI_IONIQ_5",   # Ioniq 5 N performance trim
    "MOCK": "MOCK",
    "Unknown": "UNKNOWN",
}


def canon(name):
    """Return the nameplate-level canonical model for a raw fingerprint directory name."""
    if name in SPECIAL:
        return SPECIAL[name]
    toks = name.replace("-", "_").split("_")
    brand, rest = toks[0].upper(), toks[1:]
    if brand == "ALFA":                       # two-word brand
        brand, rest = "ALFA_ROMEO", toks[2:]
    keep = [t.upper() for t in rest if t
            and not YEAR.match(t.upper())
            and not GENRE.match(t.upper())
            and t.upper() not in STRIP]
    return "_".join([brand] + keep) if keep else brand
