"""
normalize.py — All normalization functions used at Parquet conversion time.

The same normalization is applied to train S1/S2/S3 and test S1/S2/S3.
Nothing here is split-specific.

Key outputs per entity:
  name_norm     : lowercase NFKC, accent-stripped, punctuation normalized, whitespace stripped
  name_compact  : name_norm with all spaces and punctuation removed
  address_norm  : lowercase NFKC, standard abbreviation substitution, whitespace normalized
  country_norm  : trimmed + uppercased country string (open-set, no hardcoding)
  name_tokens   : sorted space-split tokens of name_norm (list stored as comma-sep string)
  missing_name  : 1 if business_name was empty/null, else 0
  missing_address: 1 if business_address was empty/null, else 0
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Character translation tables (built once at import)
# ---------------------------------------------------------------------------
# Punctuation characters to replace with a space
_PUNCT_TO_SPACE = str.maketrans(
    r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""",
    " " * len(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
)

# ---------------------------------------------------------------------------
# Address abbreviation mapping (open-set approach; additive, not exhaustive)
# Applied AFTER lowercasing and NFKC normalization.
# Using whole-word regex substitution to avoid partial matches.
# ---------------------------------------------------------------------------
_ADDR_ABBREVS = [
    (r'\bstreet\b',    'st'),
    (r'\broad\b',      'rd'),
    (r'\bavenue\b',    'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\bdrive\b',     'dr'),
    (r'\blane\b',      'ln'),
    (r'\bhighway\b',   'hwy'),
    (r'\bapartment\b', 'apt'),
    (r'\bsuite\b',     'ste'),
]
_ADDR_ABBREV_RE = [(re.compile(p), r) for p, r in _ADDR_ABBREVS]

# Collapse multiple whitespace
_WS_RE = re.compile(r'\s+')

# Digits regex (for extracting house numbers / numeric tokens)
_DIGIT_RE = re.compile(r'\d+')


# ---------------------------------------------------------------------------
# Core text normalizer
# ---------------------------------------------------------------------------
def _nfkc_strip_accents(text: str) -> str:
    """Apply NFKC + strip combining diacritical marks where safe."""
    # NFKC normalizes ligatures, full-width chars, etc.
    text = unicodedata.normalize("NFKC", text)
    # Decompose then drop combining marks (accents) — safe for Latin/Cyrillic
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def normalize_name(raw: str) -> str:
    """
    Normalize a business name string.
    Returns '' for None/empty input (missing_name flag should be set).
    """
    if not raw or not raw.strip():
        return ""
    s = raw.strip()
    s = _nfkc_strip_accents(s)
    s = s.lower()
    s = s.translate(_PUNCT_TO_SPACE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def name_compact(name_norm: str) -> str:
    """Remove all spaces from name_norm — used for exact-match blocking key."""
    return name_norm.replace(" ", "")


def normalize_address(raw: str) -> str:
    """
    Normalize a business address string.
    Returns '' for None/empty input.
    """
    if not raw or not raw.strip():
        return ""
    s = raw.strip()
    s = _nfkc_strip_accents(s)
    s = s.lower()
    s = s.translate(_PUNCT_TO_SPACE)
    # Apply address abbreviations
    for pattern, replacement in _ADDR_ABBREV_RE:
        s = pattern.sub(replacement, s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_country(raw: str) -> str:
    """
    Normalize country: trim + uppercase.
    Open-set: works for any country string without hardcoding.
    """
    if not raw or not raw.strip():
        return ""
    return raw.strip().upper()


def tokenize_name(name_norm: str) -> list:
    """Split normalized name into sorted tokens (for blocking keys)."""
    return sorted(name_norm.split()) if name_norm else []


def tokenize_address(address_norm: str) -> list:
    """Split normalized address into tokens."""
    return address_norm.split() if address_norm else []


def extract_numeric_tokens(text_norm: str) -> list:
    """Extract all numeric token sequences from a normalized string."""
    return _DIGIT_RE.findall(text_norm)


def first_house_number(address_norm: str) -> str:
    """Return the first numeric token in an address (house number proxy)."""
    nums = _DIGIT_RE.findall(address_norm)
    return nums[0] if nums else ""


def consonant_skeleton(name_compact: str, max_len: int = 16) -> str:
    """
    Remove vowels from name_compact to create a consonant skeleton.
    Useful as a blocking key robust to vowel transcription errors.
    Keeps digits. Truncated to max_len chars.
    """
    vowels = set("aeiouAEIOU")
    skeleton = "".join(c for c in name_compact if c not in vowels)
    return skeleton[:max_len]


# ---------------------------------------------------------------------------
# Vectorized pandas-friendly wrappers
# (used in Parquet conversion script via .apply() — only called once at
#  conversion time, not in the hot prediction path)
# ---------------------------------------------------------------------------
def apply_name_norm(series):
    return series.fillna("").apply(normalize_name)

def apply_name_compact(name_norm_series):
    return name_norm_series.apply(name_compact)

def apply_address_norm(series):
    return series.fillna("").apply(normalize_address)

def apply_country_norm(series):
    return series.fillna("").apply(normalize_country)

def apply_name_tokens_str(name_norm_series):
    """Return comma-separated sorted tokens (storable in Parquet as string)."""
    return name_norm_series.apply(
        lambda n: ",".join(tokenize_name(n)) if n else ""
    )

def apply_address_tokens_str(addr_norm_series):
    return addr_norm_series.apply(
        lambda a: ",".join(tokenize_address(a)) if a else ""
    )

def apply_consonant_skeleton(name_compact_series):
    return name_compact_series.apply(consonant_skeleton)

def apply_house_number(addr_norm_series):
    return addr_norm_series.apply(first_house_number)
