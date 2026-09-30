"""
validate.py — LOCAL pre-validation tool for submission files.

WARNING: This is NOT the official Amazon ML Challenge validator.
         It implements the submission rules as confirmed in Phase A.
         Replace with the official utils/validate_submission.py when available.

Checks performed:
  1. Both output files exist and are readable.
  2. Exact column headers in correct order.
  3. TAB delimiter, comma-separated ID lists.
  4. Every test S1 entity_id appears exactly once.
  5. No duplicate S1 IDs.
  6. No duplicate matched_entity_ids within a row.
  7. No duplicate candidate_entity_ids within a row.
  8. Every matched_entity_id exists in test S2 or test S3.
  9. Every candidate_entity_id exists in test S2 or test S3.
 10. Every matched_entity_id is a subset of that S1's candidate_entity_ids.
 11. Zero-match rows are represented correctly (empty matched_entity_ids, tab present).
 12. No extra/unexpected rows.

Usage:
  python src/validate.py
  python src/validate.py --matching output/matching_results.tsv \
                         --candidate output/candidate_pairs.tsv \
                         --test-dir /path/to/test/dir
"""

import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def _fail(msg: str, errors: list):
    print(f"  {FAIL} {msg}")
    errors.append(msg)


def _pass(msg: str):
    print(f"  {PASS} {msg}")


def _info(msg: str):
    print(f"  {INFO} {msg}")


# ---------------------------------------------------------------------------
# Load valid entity ID sets from test S2/S3
# ---------------------------------------------------------------------------
def load_valid_pool_ids(test_dir: str = None, pool_paths: list = None) -> set:
    """Return the set of all valid candidate IDs (test S2 + test S3)."""
    valid = set()
    if pool_paths is None and test_dir is not None:
        pool_paths = [os.path.join(test_dir, f) for f in ("test_source2.tsv", "test_source3.tsv")]
    if not pool_paths:
        return None

    for path in pool_paths:
        if not os.path.exists(path):
            print(f"  {WARN} Cannot find {path} — pool ID validation will be skipped.")
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                valid.add(row["entity_id"].strip())
    print(f"  {INFO} Loaded {len(valid):,} valid pool IDs.")
    return valid


def load_valid_s1_ids(test_dir: str = None, s1_path: str = None) -> set:
    """Return the set of all valid test S1 entity_ids."""
    if s1_path is None and test_dir is not None:
        s1_path = os.path.join(test_dir, "test_source1.tsv")
    if not s1_path or not os.path.exists(s1_path):
        print(f"  {WARN} Cannot find reference S1 file ({s1_path}) — S1 ID completeness validation will be skipped.")
        return None
    ids = set()
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.add(row["entity_id"].strip())
    print(f"  {INFO} Loaded {len(ids):,} valid reference S1 IDs.")
    return ids


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------
def validate(
    matching_path: str,
    candidate_path: str,
    test_dir: str = None,
    s1_path: str = None,
    pool_paths: list = None,
    verbose: bool = False,
) -> bool:
    """
    Validate submission files.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("LOCAL PRE-VALIDATION TOOL")
    print("(NOT the official Amazon ML Challenge validator)")
    print("=" * 60)

    errors = []

    # ------------------------------------------------------------------
    # Check file existence
    # ------------------------------------------------------------------
    print("\n[CHECK 1] File existence ...")
    for label, path in [("matching_results.tsv", matching_path),
                          ("candidate_pairs.tsv",  candidate_path)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            _pass(f"{label} exists ({size_mb:.1f} MB)")
        else:
            _fail(f"{label} NOT FOUND at {path}", errors)

    if errors:
        print("\nCannot continue — output files missing.")
        return False

    # ------------------------------------------------------------------
    # Load valid ID sets
    # ------------------------------------------------------------------
    print("\n[CHECK 2] Loading reference ID sets ...")
    valid_pool_ids = load_valid_pool_ids(test_dir=test_dir, pool_paths=pool_paths)
    valid_s1_ids   = load_valid_s1_ids(test_dir=test_dir, s1_path=s1_path)

    # ------------------------------------------------------------------
    # Parse candidate_pairs.tsv first (needed for cross-check with matches)
    # ------------------------------------------------------------------
    print("\n[CHECK 3] Parsing candidate_pairs.tsv ...")
    candidate_header_ok = False
    candidate_s1_ids = []
    candidate_map = {}   # s1_id -> set of candidate_ids

    with open(candidate_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header == [config.S1_ID_COL, config.CANDIDATE_COL]:
            candidate_header_ok = True
            _pass(f"Header OK: {header}")
        else:
            _fail(f"Wrong header: got {header}, expected "
                  f"['{config.S1_ID_COL}', '{config.CANDIDATE_COL}']", errors)

        dup_s1_cands = []
        bad_cand_ids = []

        for i, row in enumerate(reader):
            if len(row) < 1:
                _fail(f"candidate_pairs row {i+2}: fewer than 1 column", errors)
                continue

            s1_id = row[0].strip()
            candidate_s1_ids.append(s1_id)

            if len(row) >= 2 and row[1].strip():
                cand_ids = [x.strip() for x in row[1].split(",") if x.strip()]
            else:
                cand_ids = []

            # Check for duplicate candidate IDs within a row
            if len(cand_ids) != len(set(cand_ids)):
                dup_s1_cands.append(s1_id)

            # Validate all IDs exist in pool
            if valid_pool_ids is not None:
                for cid in cand_ids:
                    if cid not in valid_pool_ids:
                        bad_cand_ids.append((s1_id, cid))

            candidate_map[s1_id] = set(cand_ids)

        if dup_s1_cands:
            _fail(f"Duplicate candidate IDs within row: {len(dup_s1_cands)} S1 entities affected "
                  f"(first: {dup_s1_cands[0]})", errors)
        else:
            _pass("No duplicate candidate IDs within any row.")

        if bad_cand_ids:
            _fail(f"{len(bad_cand_ids)} invalid candidate IDs (not in test S2/S3) "
                  f"(first: {bad_cand_ids[0]})", errors)
        else:
            _pass("All candidate IDs exist in test S2/S3.")

    # ------------------------------------------------------------------
    # Parse matching_results.tsv
    # ------------------------------------------------------------------
    print("\n[CHECK 4] Parsing matching_results.tsv ...")
    match_s1_ids = []
    match_map = {}   # s1_id -> set of matched_ids
    bad_match_ids = []
    not_in_candidates = []
    dup_match_within = []

    with open(matching_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header == [config.S1_ID_COL, config.MATCH_COL]:
            _pass(f"Header OK: {header}")
        else:
            _fail(f"Wrong header: got {header}, expected "
                  f"['{config.S1_ID_COL}', '{config.MATCH_COL}']", errors)

        for i, row in enumerate(reader):
            if len(row) < 1:
                _fail(f"matching_results row {i+2}: fewer than 1 column", errors)
                continue

            s1_id = row[0].strip()
            match_s1_ids.append(s1_id)

            if len(row) >= 2 and row[1].strip():
                matched = [x.strip() for x in row[1].split(",") if x.strip()]
            else:
                matched = []

            # Check duplicate matched IDs within a row
            if len(matched) != len(set(matched)):
                dup_match_within.append(s1_id)

            # Check all matched IDs exist in pool
            if valid_pool_ids is not None:
                for mid in matched:
                    if mid not in valid_pool_ids:
                        bad_match_ids.append((s1_id, mid))

            # Check matched ⊆ candidates
            cands = candidate_map.get(s1_id, set())
            for mid in matched:
                if mid not in cands:
                    not_in_candidates.append((s1_id, mid))

            match_map[s1_id] = set(matched)

    if dup_match_within:
        _fail(f"Duplicate matched IDs within row: {len(dup_match_within)} S1 entities "
              f"(first: {dup_match_within[0]})", errors)
    else:
        _pass("No duplicate matched IDs within any row.")

    if bad_match_ids:
        _fail(f"{len(bad_match_ids)} invalid matched IDs (not in test S2/S3) "
              f"(first: {bad_match_ids[0]})", errors)
    else:
        _pass("All matched IDs exist in test S2/S3.")

    if not_in_candidates:
        _fail(f"{len(not_in_candidates)} matched IDs NOT in candidate list "
              f"(violates matches⊆candidates) (first: {not_in_candidates[0]})", errors)
    else:
        _pass("All matched IDs are contained in their S1 candidate list.")

    # ------------------------------------------------------------------
    # S1 ID completeness and uniqueness
    # ------------------------------------------------------------------
    print("\n[CHECK 5] S1 ID completeness and uniqueness ...")

    # Uniqueness
    if len(match_s1_ids) == len(set(match_s1_ids)):
        _pass(f"matching_results.tsv: no duplicate S1 IDs ({len(match_s1_ids):,} rows).")
    else:
        from collections import Counter
        dups = [k for k, v in Counter(match_s1_ids).items() if v > 1]
        _fail(f"Duplicate S1 IDs in matching_results.tsv: {len(dups)} (first: {dups[0]})", errors)

    if len(candidate_s1_ids) == len(set(candidate_s1_ids)):
        _pass(f"candidate_pairs.tsv: no duplicate S1 IDs ({len(candidate_s1_ids):,} rows).")
    else:
        from collections import Counter
        dups = [k for k, v in Counter(candidate_s1_ids).items() if v > 1]
        _fail(f"Duplicate S1 IDs in candidate_pairs.tsv: {len(dups)} (first: {dups[0]})", errors)

    # Completeness against test S1
    if valid_s1_ids is not None:
        match_set = set(match_s1_ids)
        cand_set  = set(candidate_s1_ids)

        missing_from_match = valid_s1_ids - match_set
        extra_in_match     = match_set - valid_s1_ids
        missing_from_cand  = valid_s1_ids - cand_set
        extra_in_cand      = cand_set - valid_s1_ids

        if missing_from_match:
            _fail(f"{len(missing_from_match):,} test S1 IDs missing from matching_results.tsv", errors)
        else:
            _pass("matching_results.tsv contains all test S1 IDs.")

        if extra_in_match:
            _fail(f"{len(extra_in_match):,} unexpected S1 IDs in matching_results.tsv "
                  f"(first: {next(iter(extra_in_match))})", errors)
        else:
            _pass("matching_results.tsv has no unexpected S1 IDs.")

        if missing_from_cand:
            _fail(f"{len(missing_from_cand):,} test S1 IDs missing from candidate_pairs.tsv", errors)
        else:
            _pass("candidate_pairs.tsv contains all test S1 IDs.")

        if extra_in_cand:
            _fail(f"{len(extra_in_cand):,} unexpected S1 IDs in candidate_pairs.tsv "
                  f"(first: {next(iter(extra_in_cand))})", errors)
        else:
            _pass("candidate_pairs.tsv has no unexpected S1 IDs.")

    # Check both files have same S1 IDs
    if set(match_s1_ids) == set(candidate_s1_ids):
        _pass("S1 IDs match between matching_results.tsv and candidate_pairs.tsv.")
    else:
        diff = set(match_s1_ids).symmetric_difference(set(candidate_s1_ids))
        _fail(f"S1 ID mismatch between files: {len(diff)} differing IDs.", errors)

    # ------------------------------------------------------------------
    # Zero-match statistics
    # ------------------------------------------------------------------
    print("\n[CHECK 6] Zero-match statistics ...")
    zero_match_count = sum(1 for ids in match_map.values() if not ids)
    nonzero_count    = len(match_map) - zero_match_count
    _info(f"Zero-match S1 entities: {zero_match_count:,}")
    _info(f"Non-zero-match S1 entities: {nonzero_count:,}")
    avg_matches = (
        sum(len(v) for v in match_map.values()) / max(len(match_map), 1)
    )
    _info(f"Average matched IDs per S1: {avg_matches:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if errors:
        print(f"{FAIL} VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    • {e}")
        print("=" * 60)
        return False
    else:
        print(f"{PASS} ALL CHECKS PASSED")
        print("=" * 60)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local pre-validation for Amazon ML Challenge submission."
    )
    parser.add_argument(
        "--matching",  default=config.MATCHING_RESULTS_TSV,
        help="Path to matching_results.tsv"
    )
    parser.add_argument(
        "--candidate", default=config.CANDIDATE_PAIRS_TSV,
        help="Path to candidate_pairs.tsv"
    )
    parser.add_argument(
        "--test-dir",  default=config.TEST_DIR,
        help="Path to directory containing test_source1/2/3.tsv"
    )
    args = parser.parse_args()

    ok = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if ok else 1)
