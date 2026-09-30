"""
predict.py — Chunked test prediction pipeline.

Processes TEST Source-1 in chunks of PREDICT_CHUNK_SIZE rows.
For each chunk:
  1. Read S1 rows from Parquet
  2. Generate blocking candidates (10 views via DuckDB)
  3. Join S1 + pool columns
  4. Compute 43 features
  5. predict_proba in batches
  6. Apply selected threshold
  7. Write chunk output to Parquet incrementally

Final assembly:
  - Reads all chunk Parquet files via DuckDB
  - Aggregates candidate IDs and matched IDs per S1
  - Writes matching_results.tsv and candidate_pairs.tsv
  - Retains all test S1 IDs (zero-match rows preserved)

Usage:
  python src/predict.py
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import joblib

import config
import features as F
from blocking import run_blocking, get_con

config.ensure_dirs()

CHUNK_PARQUET_DIR = os.path.join(config.PARQUET_DIR, "pred_chunks")
os.makedirs(CHUNK_PARQUET_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load the saved model artifact
# ---------------------------------------------------------------------------
def load_artifact():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run model.py first."
        )
    return joblib.load(config.MODEL_PATH)


# ---------------------------------------------------------------------------
# Join S1 chunk with pool candidates to get all feature columns
# ---------------------------------------------------------------------------
def join_s1_pool(
    candidates_df: pd.DataFrame,
    s1_chunk_df: pd.DataFrame,
    pool_parquet: str,
    con=None,
) -> pd.DataFrame:
    """
    Join candidate pair IDs with S1 chunk and pool data to get feature columns.
    Uses DuckDB for the pool join (avoids loading full pool into RAM).
    """
    close_con = con is None
    if con is None:
        con = get_con()

    # Register dataframes as DuckDB tables
    con.register("_cand_pairs", candidates_df[["source1_entity_id", "candidate_entity_id", "src",
                                                 "n_views", "name_jaccard",
                                                 "hit_V1","hit_V2","hit_V3","hit_V4","hit_V5",
                                                 "hit_V6","hit_V7","hit_V8","hit_V9","hit_V10"]])
    con.register("_s1_chunk", s1_chunk_df[[
        "entity_id", "name_norm", "name_compact", "name_tokens",
        "address_norm", "country_norm", "missing_name", "missing_address"
    ]])

    pool = pool_parquet.replace("\\", "/")

    result = con.execute(f"""
        SELECT
            cp.source1_entity_id,
            cp.candidate_entity_id,
            cp.src,
            cp.n_views,
            cp.name_jaccard,
            cp.hit_V1, cp.hit_V2, cp.hit_V3, cp.hit_V4, cp.hit_V5,
            cp.hit_V6, cp.hit_V7, cp.hit_V8, cp.hit_V9, cp.hit_V10,
            -- S1 columns
            s1.name_norm        AS s1_name_norm,
            s1.name_compact     AS s1_name_compact,
            s1.name_tokens      AS s1_name_tokens,
            s1.address_norm     AS s1_address_norm,
            s1.country_norm     AS s1_country_norm,
            s1.missing_name     AS s1_missing_name,
            s1.missing_address  AS s1_missing_address,
            -- Candidate (pool) columns
            p.name_norm         AS cand_name_norm,
            p.name_compact      AS cand_name_compact,
            p.name_tokens       AS cand_name_tokens,
            p.address_norm      AS cand_address_norm,
            p.country_norm      AS cand_country_norm,
            p.missing_name      AS cand_missing_name,
            p.missing_address   AS cand_missing_address
        FROM _cand_pairs cp
        JOIN _s1_chunk s1 ON s1.entity_id = cp.source1_entity_id
        JOIN '{pool}' p   ON p.entity_id  = cp.candidate_entity_id
    """).df()

    con.unregister("_cand_pairs")
    con.unregister("_s1_chunk")

    if close_con:
        con.close()

    return result


# ---------------------------------------------------------------------------
# Process one chunk of S1 rows
# ---------------------------------------------------------------------------
def process_chunk(
    s1_chunk_df: pd.DataFrame,
    pool_parquet: str,
    token_df_parquet: str,
    artifact: dict,
    chunk_idx: int,
    benchmark_mode: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline for one S1 chunk.
    Returns a DataFrame with: source1_entity_id, candidate_entity_id, src,
                               score, is_match (0/1), n_views, hit_* cols.
    """
    t0 = time.time()
    n_s1 = len(s1_chunk_df)
    print(f"\n  Chunk {chunk_idx}: {n_s1:,} S1 rows")

    # Write chunk to temp parquet for DuckDB to scan
    chunk_s1_path = os.path.join(CHUNK_PARQUET_DIR, f"_s1_chunk_{chunk_idx}.parquet")
    s1_chunk_df.to_parquet(chunk_s1_path, engine="pyarrow", index=False)

    # 1. Generate blocking candidates
    t1 = time.time()
    candidates_df = run_blocking(
        s1_parquet=chunk_s1_path,
        pool_parquet=pool_parquet,
        token_df_parquet=token_df_parquet,
        max_block_size=config.MAX_BLOCK_SIZE,
        top_k=config.TOP_K,
        verbose=False,
    )
    t_block = time.time() - t1
    n_cands = len(candidates_df)
    print(f"    Blocking: {n_cands:,} candidates in {t_block:.1f}s "
          f"(avg {n_cands/max(n_s1,1):.1f}/S1)")

    if n_cands == 0:
        print(f"    No candidates found for this chunk.")
        return pd.DataFrame(columns=[
            "source1_entity_id","candidate_entity_id","src","score","is_match","n_views"
        ])

    # 2. Join with S1 + pool data
    t2 = time.time()
    joined_df = join_s1_pool(candidates_df, s1_chunk_df, pool_parquet)
    t_join = time.time() - t2
    print(f"    Join: {len(joined_df):,} rows in {t_join:.1f}s")

    # 3. Compute features
    t3 = time.time()
    X = F.compute_features(joined_df)
    t_feat = time.time() - t3
    print(f"    Features: {X.shape} in {t_feat:.1f}s")

    # 4. predict_proba in batches of 50K to limit RAM
    t4 = time.time()
    model = artifact["model"]
    batch = 50_000
    scores = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch):
        end = min(start + batch, len(X))
        scores[start:end] = model.predict_proba(X[start:end])[:, 1]
    t_pred = time.time() - t4

    # 5. Apply threshold
    threshold = artifact["threshold"]
    is_match  = (scores >= threshold).astype(np.int8)
    n_matches = is_match.sum()
    print(f"    Predict: {t_pred:.1f}s | threshold={threshold:.3f} | "
          f"matches={n_matches:,}/{n_cands:,}")

    # 6. Build result DataFrame
    result = pd.DataFrame({
        "source1_entity_id":  joined_df["source1_entity_id"].values,
        "candidate_entity_id": joined_df["candidate_entity_id"].values,
        "src":                 joined_df["src"].values,
        "score":               scores,
        "is_match":            is_match,
        "n_views":             joined_df["n_views"].values.astype(np.int8),
    })

    chunk_elapsed = time.time() - t0
    print(f"    Chunk total: {chunk_elapsed:.1f}s")

    if benchmark_mode:
        full_s1_count = 1_732_544
        chunks_needed = full_s1_count / max(n_s1, 1)
        eta_minutes = (chunk_elapsed * chunks_needed) / 60
        print(f"\n  === BENCHMARK ESTIMATE ===")
        print(f"  Chunk size: {n_s1:,}")
        print(f"  Chunk time: {chunk_elapsed:.1f}s")
        print(f"  Chunks needed for full test: {chunks_needed:.0f}")
        print(f"  Estimated full-test runtime: {eta_minutes:.0f} minutes")
        print(f"  ===========================")

    # Cleanup temp chunk parquet
    try:
        os.remove(chunk_s1_path)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Full test prediction pipeline
# ---------------------------------------------------------------------------
def run_full_prediction(benchmark_first_chunk: bool = True):
    """
    Process all test S1 in chunks.
    Writes incremental Parquet; assembles TSV outputs at the end.
    """
    print("=" * 60)
    print("PHASE I/K — Full Test Prediction")
    print("=" * 60)

    artifact = load_artifact()
    print(f"  Model: {artifact['model_name']}")
    print(f"  Threshold: {artifact['threshold']:.3f}")
    print(f"  OOF F0.5: {artifact['oof_f05']:.4f}")

    # Read test S1 (only columns needed for blocking + joining)
    print(f"\n  Loading test S1 index ...")
    s1_pq = pq.ParquetFile(config.TEST_S1_PARQUET)
    total_s1 = s1_pq.metadata.num_rows
    print(f"  Total test S1: {total_s1:,}")

    chunk_size = config.PREDICT_CHUNK_SIZE
    n_chunks   = (total_s1 + chunk_size - 1) // chunk_size

    all_chunk_files = []
    total_start = time.time()

    for chunk_idx in range(n_chunks):
        chunk_path = os.path.join(CHUNK_PARQUET_DIR, f"chunk_{chunk_idx:04d}.parquet")

        if os.path.exists(chunk_path):
            print(f"\n  Chunk {chunk_idx}: [SKIP] already exists.")
            all_chunk_files.append(chunk_path)
            continue

        # Read S1 chunk
        start_row = chunk_idx * chunk_size
        s1_chunk_df = s1_pq.read_row_groups(
            # Use DuckDB for offset+limit to avoid reading full file into pandas
        ).to_pandas() if False else None

        # Cleaner approach: use DuckDB offset/limit
        con_tmp = get_con()
        s1p = config.TEST_S1_PARQUET.replace("\\", "/")
        s1_chunk_df = con_tmp.execute(
            f"SELECT * FROM '{s1p}' LIMIT {chunk_size} OFFSET {start_row}"
        ).df()
        con_tmp.close()

        if len(s1_chunk_df) == 0:
            break

        is_first = (chunk_idx == 0)
        result_df = process_chunk(
            s1_chunk_df,
            config.TEST_POOL_PARQUET,
            config.TEST_TOKEN_DF_PARQUET,
            artifact,
            chunk_idx,
            benchmark_mode=(is_first and benchmark_first_chunk),
        )

        if len(result_df) > 0:
            result_df.to_parquet(chunk_path, engine="pyarrow", index=False)
        all_chunk_files.append(chunk_path)

        if is_first and benchmark_first_chunk:
            answer = input("\n  Continue with remaining chunks? [y/N]: ").strip().lower()
            if answer != "y":
                print("  Stopping after benchmark chunk.")
                return

    total_elapsed = time.time() - total_start
    print(f"\n  All chunks complete in {total_elapsed/60:.1f} minutes.")

    # Assemble final output files
    assemble_outputs(all_chunk_files)


# ---------------------------------------------------------------------------
# Assemble final TSV outputs
# ---------------------------------------------------------------------------
def assemble_outputs(chunk_files: list):
    """
    Read all chunk Parquets, aggregate per S1, write TSV output files.
    Uses DuckDB to aggregate without loading all rows into RAM.
    """
    print("\n  Assembling final outputs ...")

    # Write list of chunk files for DuckDB to read
    existing_chunks = [f for f in chunk_files if os.path.exists(f)]
    if not existing_chunks:
        print("  No chunk files found — skipping assembly.")
        return

    con = get_con()

    # Create a glob pattern or union of chunk files
    chunk_glob = os.path.join(CHUNK_PARQUET_DIR, "chunk_*.parquet").replace("\\", "/")
    s1_parquet = config.TEST_S1_PARQUET.replace("\\", "/")

    # Aggregate candidates per S1 (all predictions, regardless of is_match)
    print("  Aggregating candidate_pairs ...")
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _all_results AS
        SELECT * FROM '{chunk_glob}'
    """)

    # Candidate pairs: all candidates scored
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _cand_agg AS
        SELECT
            source1_entity_id,
            STRING_AGG(DISTINCT candidate_entity_id, ',') AS candidate_entity_ids
        FROM _all_results
        GROUP BY source1_entity_id
    """)

    # Match results: only candidates above threshold (is_match=1)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _match_agg AS
        SELECT
            source1_entity_id,
            STRING_AGG(DISTINCT candidate_entity_id, ',') AS matched_entity_ids
        FROM _all_results
        WHERE is_match = 1
        GROUP BY source1_entity_id
    """)

    # Master S1 list — ensures every test S1 appears
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _s1_master AS
        SELECT entity_id AS source1_entity_id FROM '{s1_parquet}'
    """)

    # Write matching_results.tsv
    print("  Writing matching_results.tsv ...")
    match_out = config.MATCHING_RESULTS_TSV.replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT
                m.source1_entity_id,
                COALESCE(r.matched_entity_ids, '') AS matched_entity_ids
            FROM _s1_master m
            LEFT JOIN _match_agg r ON r.source1_entity_id = m.source1_entity_id
            ORDER BY m.source1_entity_id
        ) TO '{match_out}'
        (FORMAT CSV, DELIMITER '\t', HEADER TRUE)
    """)

    # Write candidate_pairs.tsv
    print("  Writing candidate_pairs.tsv ...")
    cand_out = config.CANDIDATE_PAIRS_TSV.replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT
                m.source1_entity_id,
                COALESCE(c.candidate_entity_ids, '') AS candidate_entity_ids
            FROM _s1_master m
            LEFT JOIN _cand_agg c ON c.source1_entity_id = m.source1_entity_id
            ORDER BY m.source1_entity_id
        ) TO '{cand_out}'
        (FORMAT CSV, DELIMITER '\t', HEADER TRUE)
    """)

    con.close()

    # Report sizes
    for label, path in [("matching_results.tsv", config.MATCHING_RESULTS_TSV),
                         ("candidate_pairs.tsv",  config.CANDIDATE_PAIRS_TSV)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {label}: {size_mb:.1f} MB")

    print("\n  Output assembly complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip benchmark pause after first chunk")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Only assemble TSVs from existing chunk Parquets")
    args = parser.parse_args()

    if args.assemble_only:
        import glob
        chunk_files = sorted(glob.glob(
            os.path.join(CHUNK_PARQUET_DIR, "chunk_*.parquet")
        ))
        assemble_outputs(chunk_files)
    else:
        run_full_prediction(benchmark_first_chunk=not args.no_benchmark)
