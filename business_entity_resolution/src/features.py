"""
features.py — 43-feature extractor for candidate pairs.

The SAME feature definitions are used for training, validation, and test prediction.
No split-specific logic here.

Input:  DataFrame with columns from blocking.py +
        joined S1 and candidate (pool) row data.
Output: float32 numpy array + ordered feature name list.

Feature groups:
  NAME (1-11), ADDRESS (12-19), COUNTRY (20), MISSINGNESS (21-24),
  BLOCKING EVIDENCE (25-36), RARITY (37-39), INTERACTIONS (40-43).
"""

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False
    print("[WARN] jellyfish not available — Jaro-Winkler/Levenshtein will be 0.0")


# ---------------------------------------------------------------------------
# Feature name list (canonical ordering — must never change between runs)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # NAME
    "f01_name_exact",
    "f02_name_compact_exact",
    "f03_name_sorted_tokens_exact",
    "f04_name_token_jaccard",
    "f05_name_token_overlap_min",
    "f06_name_first_token_match",
    "f07_name_levenshtein_sim",
    "f08_name_jaro_winkler",
    "f09_name_trigram_jaccard",
    "f10_name_len_ratio",
    "f11_name_char_sim",
    # ADDRESS
    "f12_addr_exact",
    "f13_addr_token_jaccard",
    "f14_addr_token_overlap_min",
    "f15_addr_levenshtein_sim",
    "f16_addr_trigram_jaccard",
    "f17_addr_numeric_jaccard",
    "f18_addr_both_numeric_present",
    "f19_addr_last_numeric_match",
    # COUNTRY
    "f20_country_match",
    # MISSINGNESS
    "f21_s1_missing_name",
    "f22_cand_missing_name",
    "f23_s1_missing_address",
    "f24_cand_missing_address",
    # BLOCKING EVIDENCE
    "f25_n_views",
    "f26_hit_V1",
    "f27_hit_V2",
    "f28_hit_V3",
    "f29_hit_V4",
    "f30_hit_V5",
    "f31_hit_V6",
    "f32_hit_V7",
    "f33_hit_V8",
    "f34_hit_V9",
    "f35_hit_V10",
    "f36_src_is_S3",
    # RARITY
    "f37_log_name_freq_pool",
    "f38_log_name_freq_s1",
    "f39_log_candidate_count",
    # INTERACTIONS
    "f40_name_x_addr_sim",
    "f41_strong_name_weak_addr",
    "f42_weak_name_strong_addr",
    "f43_country_x_name_sim",
]

N_FEATURES = len(FEATURE_NAMES)  # 43


# ---------------------------------------------------------------------------
# Sub-utilities
# ---------------------------------------------------------------------------

def _trigrams(s: str) -> set:
    """Character trigram set."""
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _token_overlap_min(a: set, b: set) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _lev_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0,1]."""
    if not HAS_JELLYFISH:
        return 0.0
    if s1 == s2:
        return 1.0
    d = jellyfish.levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - d / max_len if max_len > 0 else 1.0


def _jw(s1: str, s2: str) -> float:
    if not HAS_JELLYFISH:
        return 0.0
    if not s1 or not s2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)


def _char_sim(s1: str, s2: str) -> float:
    """Character bigram Jaccard."""
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}
    return _jaccard(bigrams(s1), bigrams(s2))


def _numeric_tokens(s: str) -> set:
    import re
    return set(re.findall(r'\d+', s))


def _last_numeric(s: str):
    import re
    nums = re.findall(r'\d+', s)
    return nums[-1] if nums else None


# ---------------------------------------------------------------------------
# Main feature computation (vectorized over DataFrame rows)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute all 43 features for a DataFrame of candidate pairs.

    Required columns (from the joined S1 + pool data):
        s1_name_norm, s1_name_compact, s1_name_tokens (comma-sep str),
        s1_address_norm, s1_country_norm,
        s1_missing_name (int8), s1_missing_address (int8),
        cand_name_norm, cand_name_compact, cand_name_tokens,
        cand_address_norm, cand_country_norm,
        cand_missing_name, cand_missing_address,
        n_views, hit_V1..hit_V10, src,
        name_jaccard (from blocking),
        [optional] log_name_freq_pool, log_name_freq_s1, log_candidate_count

    Returns: float32 numpy array of shape (n_rows, N_FEATURES)
    """
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Pre-extract numpy arrays for fast access — avoids .iloc[i] in the hot loop
    s1_nn    = df["s1_name_norm"].fillna("").values
    cand_nn  = df["cand_name_norm"].fillna("").values
    s1_nc    = df["s1_name_compact"].fillna("").values
    cand_nc  = df["cand_name_compact"].fillna("").values
    s1_nt    = df["s1_name_tokens"].fillna("").values
    cand_nt  = df["cand_name_tokens"].fillna("").values
    s1_an    = df["s1_address_norm"].fillna("").values
    cand_an  = df["cand_address_norm"].fillna("").values
    s1_cn    = df["s1_country_norm"].fillna("").values
    cand_cn  = df["cand_country_norm"].fillna("").values
    # Missingness as float32 arrays — extracted BEFORE the loop
    s1_miss_name    = df["s1_missing_name"].fillna(0).values.astype(np.float32)
    cand_miss_name  = df["cand_missing_name"].fillna(0).values.astype(np.float32)
    s1_miss_addr    = df["s1_missing_address"].fillna(0).values.astype(np.float32)
    cand_miss_addr  = df["cand_missing_address"].fillna(0).values.astype(np.float32)

    for i in range(n):
        sn  = s1_nn[i];   cn  = cand_nn[i]
        snc = s1_nc[i];   cnc = cand_nc[i]
        snt = set(s1_nt[i].split(",")) - {""} if s1_nt[i] else set()
        cnt = set(cand_nt[i].split(",")) - {""} if cand_nt[i] else set()
        sa  = s1_an[i];   ca  = cand_an[i]
        sc  = s1_cn[i];   cc  = cand_cn[i]

        # ---- NAME features ----
        X[i, 0] = float(sn == cn and sn != "")                          # f01 exact
        X[i, 1] = float(snc == cnc and snc != "")                        # f02 compact exact
        X[i, 2] = float(snt and cnt and sorted(snt) == sorted(cnt))      # f03 sorted tokens
        X[i, 3] = _jaccard(snt, cnt)                                      # f04 token jaccard
        X[i, 4] = _token_overlap_min(snt, cnt)                           # f05 overlap/min
        s1_first = s1_nt[i].split(",")[0] if s1_nt[i] else ""
        cn_first = cand_nt[i].split(",")[0] if cand_nt[i] else ""
        X[i, 5] = float(s1_first and s1_first == cn_first)               # f06 first token
        X[i, 6] = _lev_sim(sn, cn)                                        # f07 levenshtein
        X[i, 7] = _jw(sn, cn)                                             # f08 jaro-winkler
        X[i, 8] = _jaccard(_trigrams(sn), _trigrams(cn))                  # f09 trigram
        max_len_n = max(len(sn), len(cn))
        X[i, 9]  = min(len(sn), len(cn)) / max_len_n if max_len_n > 0 else 1.0  # f10 len ratio
        X[i, 10] = _char_sim(sn, cn)                                      # f11 char sim

        # ---- ADDRESS features ----
        X[i, 11] = float(sa == ca and sa != "")                          # f12 exact addr
        sat = set(sa.split()) - {""} if sa else set()
        cat = set(ca.split()) - {""} if ca else set()
        X[i, 12] = _jaccard(sat, cat)                                     # f13 addr token jaccard
        X[i, 13] = _token_overlap_min(sat, cat)                          # f14 addr overlap/min
        X[i, 14] = _lev_sim(sa[:80], ca[:80])                            # f15 addr levenshtein (capped)
        X[i, 15] = _jaccard(_trigrams(sa[:60]), _trigrams(ca[:60]))      # f16 addr trigram
        snum = _numeric_tokens(sa); cnum = _numeric_tokens(ca)
        X[i, 16] = _jaccard(snum, cnum)                                   # f17 numeric jaccard
        X[i, 17] = float(bool(snum) and bool(cnum))                      # f18 both numeric present
        sl = _last_numeric(sa); cl = _last_numeric(ca)
        X[i, 18] = float(sl is not None and sl == cl)                     # f19 last numeric match

        # ---- COUNTRY ----
        X[i, 19] = float(sc == cc and sc != "")                          # f20 country match

        # ---- MISSINGNESS (use pre-extracted arrays — no .iloc) ----
        X[i, 20] = s1_miss_name[i]    # f21
        X[i, 21] = cand_miss_name[i]  # f22
        X[i, 22] = s1_miss_addr[i]    # f23
        X[i, 23] = cand_miss_addr[i]  # f24

    # ---- BLOCKING EVIDENCE (vectorized) ----
    X[:, 24] = df["n_views"].values.astype(np.float32)                   # f25
    for vi, vn in enumerate(range(1, 11)):
        col = f"hit_V{vn}"
        if col in df.columns:
            X[:, 25 + vi] = df[col].values.astype(np.float32)
    X[:, 35] = (df["src"].values == "S3").astype(np.float32)             # f36 src_is_S3

    # ---- RARITY (vectorized) ----
    if "log_name_freq_pool" in df.columns:
        X[:, 36] = df["log_name_freq_pool"].fillna(0).values.astype(np.float32)
    if "log_name_freq_s1" in df.columns:
        X[:, 37] = df["log_name_freq_s1"].fillna(0).values.astype(np.float32)
    if "log_candidate_count" in df.columns:
        X[:, 38] = df["log_candidate_count"].fillna(0).values.astype(np.float32)

    # ---- INTERACTIONS (vectorized) ----
    name_sim  = np.maximum(X[:, 7], X[:, 3])   # max(jaro_winkler, token_jaccard)
    addr_sim  = np.maximum(X[:, 12], X[:, 14]) # max(addr_token_jaccard, addr_levenshtein)
    X[:, 39] = name_sim * addr_sim              # f40 name x addr
    X[:, 40] = (name_sim > 0.8).astype(np.float32) * (addr_sim < 0.3).astype(np.float32)  # f41
    X[:, 41] = (name_sim < 0.3).astype(np.float32) * (addr_sim > 0.8).astype(np.float32)  # f42
    X[:, 42] = X[:, 19] * name_sim              # f43 country x name_sim

    return X


def get_feature_names() -> list:
    return list(FEATURE_NAMES)
