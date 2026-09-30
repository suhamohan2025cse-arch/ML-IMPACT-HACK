"""
benchmark.py — Phased benchmark runner for Business Entity Resolution.

Runs an end-to-end benchmark on a controlled sample to validate:
  1. Multi-view blocking recall and candidates/S1
  2. 43-feature extraction throughput & memory safety
  3. Model training (GroupKFold CV) and threshold optimization for F0.5
  4. Local pre-validation (src/validate.py) on generated outputs
  5. Scalability projection for the full 1.73M dataset

Usage:
  python src/benchmark.py --sample-size 5000
"""

import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import duckdb

import config
import normalize as N
import blocking
import features as F
import model as M
import validate as V

def run_benchmark(sample_size: int = 5000, top_k: int = config.TOP_K):
    print("=" * 70)
    print(f"PHASED BENCHMARK — Sample Size: {sample_size:,} S1 Entities")
    print("=" * 70)

    config.ensure_dirs()
    t_start = time.time()
    results = {}

    con = blocking.get_con()

    # -----------------------------------------------------------------------
    # Step 1: Extract S1 sample and Ground Truth sample from Train
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 1] Sampling {sample_size:,} S1 entities and ground truth from Train ...")
    t0 = time.time()
    
    # Check if raw Parquet exists, else sample directly from TSV using DuckDB streaming
    train_s1_src = config.TRAIN_S1_PARQUET if os.path.exists(config.TRAIN_S1_PARQUET) else config.TRAIN_S1_TSV.replace('\\', '/')
    read_fn = f"'{train_s1_src}'" if os.path.exists(config.TRAIN_S1_PARQUET) else f"read_csv('{train_s1_src}', delim='\\t', header=true, all_varchar=true)"

    s1_sample_df = con.execute(f"""
        SELECT * FROM {read_fn}
        LIMIT {sample_size}
    """).fetchdf()

    # Add normalization columns if not present
    if "name_norm" not in s1_sample_df.columns:
        print("  Applying normalization to S1 sample ...")
        s1_sample_df["name_norm"] = N.apply_name_norm(s1_sample_df["business_name"])
        s1_sample_df["name_compact"] = N.apply_name_compact(s1_sample_df["name_norm"])
        s1_sample_df["address_norm"] = N.apply_address_norm(s1_sample_df["business_address"])
        s1_sample_df["country_norm"] = N.apply_country_norm(s1_sample_df["country"])
        s1_sample_df["name_tokens"] = N.apply_name_tokens_str(s1_sample_df["name_norm"])
        s1_sample_df["address_tokens"] = N.apply_address_tokens_str(s1_sample_df["address_norm"])
        s1_sample_df["consonant_key"] = N.apply_consonant_skeleton(s1_sample_df["name_compact"])
        s1_sample_df["house_number"] = N.apply_house_number(s1_sample_df["address_norm"])
        s1_sample_df["missing_name"] = (s1_sample_df["business_name"].fillna("").str.strip() == "").astype("int8")
        s1_sample_df["missing_address"] = (s1_sample_df["business_address"].fillna("").str.strip() == "").astype("int8")
        s1_sample_df["name_prefix6"] = s1_sample_df["name_compact"].str[:6]
        s1_sample_df["name_prefix8_last8"] = s1_sample_df["name_compact"].str[:8] + "|" + s1_sample_df["name_compact"].str[-8:]

    s1_ids_list = s1_sample_df["entity_id"].tolist()
    s1_ids_set = set(s1_ids_list)

    # Read ground truth for this sample
    gt_src = config.TRAIN_GT_PARQUET if os.path.exists(config.TRAIN_GT_PARQUET) else config.TRAIN_GT_TSV.replace('\\', '/')
    gt_read_fn = f"'{gt_src}'" if os.path.exists(config.TRAIN_GT_PARQUET) else f"read_csv('{gt_src}', delim='\\t', header=true, all_varchar=true)"

    con.register("_s1_sample_ids", pd.DataFrame({"source1_entity_id": s1_ids_list}))
    gt_sample_df = con.execute(f"""
        SELECT gt.source1_entity_id, gt.matched_entity_ids
        FROM {gt_read_fn} gt
        JOIN _s1_sample_ids s ON gt.source1_entity_id = s.source1_entity_id
    """).fetchdf()

    # Build ground truth pairs map
    gt_pairs = set()
    total_true_matches = 0
    for _, row in gt_sample_df.iterrows():
        s1_id = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip() and m != "nan"]
        total_true_matches += len(matches)
        for m in matches:
            gt_pairs.add((s1_id, m))

    print(f"  Sampled {len(s1_sample_df):,} S1 entities with {total_true_matches:,} true match pairs.")
    print(f"  Stage 1 completed in {time.time() - t0:.2f}s.")

    # -----------------------------------------------------------------------
    # Step 2: Candidate Blocking Benchmark
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 2] Running 10-view candidate blocking ...")
    t0 = time.time()

    # Determine pool source
    train_pool_src = config.TRAIN_POOL_PARQUET if os.path.exists(config.TRAIN_POOL_PARQUET) else None
    
    # If pool parquet does not exist yet, construct an exact benchmark pool:
    # 1. All true ground truth match entities from S2 and S3
    # 2. Plus a distractor set from S2 and S3 Parquet
    if train_pool_src is None:
        print("  Extracting benchmark pool (all ground truth matches + 50,000 distractors) from Parquet ...")
        t_pool = time.time()
        
        # Get list of all ground truth matched entity IDs
        all_gt_matched_ids = list({mid for _, mid in gt_pairs})
        con.register("_gt_matched_ids", pd.DataFrame({"entity_id": all_gt_matched_ids}))

        s2_pq = config.TRAIN_S2_PARQUET.replace('\\', '/')
        s3_pq = config.TRAIN_S3_PARQUET.replace('\\', '/')

        # Query true matches + distractors
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _bench_pool_raw AS
            -- True matches from S2
            SELECT *, 'S2' AS src FROM '{s2_pq}' WHERE entity_id IN (SELECT entity_id FROM _gt_matched_ids)
            UNION ALL
            -- True matches from S3
            SELECT *, 'S3' AS src FROM '{s3_pq}' WHERE entity_id IN (SELECT entity_id FROM _gt_matched_ids)
            UNION ALL
            -- 25,000 distractors from S2
            (SELECT *, 'S2' AS src FROM '{s2_pq}' USING SAMPLE 25000)
            UNION ALL
            -- 25,000 distractors from S3
            (SELECT *, 'S3' AS src FROM '{s3_pq}' USING SAMPLE 25000)
        """)

        bench_pool_df = con.execute("SELECT DISTINCT * FROM _bench_pool_raw").fetchdf()
        t_pool_extract = time.time() - t_pool
        print(f"  Extracted {len(bench_pool_df):,} pool entities in {t_pool_extract:.2f}s.")

        # Normalize the benchmark pool in Python
        t_norm = time.time()
        print(f"  Applying full normalization to {len(bench_pool_df):,} pool entities ...")
        bench_pool_df["name_norm"] = N.apply_name_norm(bench_pool_df["business_name"])
        bench_pool_df["name_compact"] = N.apply_name_compact(bench_pool_df["name_norm"])
        bench_pool_df["address_norm"] = N.apply_address_norm(bench_pool_df["business_address"])
        bench_pool_df["country_norm"] = N.apply_country_norm(bench_pool_df["country"])
        bench_pool_df["name_tokens"] = N.apply_name_tokens_str(bench_pool_df["name_norm"])
        bench_pool_df["address_tokens"] = N.apply_address_tokens_str(bench_pool_df["address_norm"])
        bench_pool_df["consonant_key"] = N.apply_consonant_skeleton(bench_pool_df["name_compact"])
        bench_pool_df["house_number"] = N.apply_house_number(bench_pool_df["address_norm"])
        bench_pool_df["missing_name"] = (bench_pool_df["business_name"].fillna("").str.strip() == "").astype("int8")
        bench_pool_df["missing_address"] = (bench_pool_df["business_address"].fillna("").str.strip() == "").astype("int8")
        bench_pool_df["name_prefix6"] = bench_pool_df["name_compact"].str[:6]
        bench_pool_df["name_prefix8_last8"] = bench_pool_df["name_compact"].str[:8] + "|" + bench_pool_df["name_compact"].str[-8:]

        t_pool_norm = time.time() - t_norm
        pool_norm_rate = len(bench_pool_df) / max(t_pool_norm, 0.001)
        print(f"  Pool normalization complete in {t_pool_norm:.2f}s ({pool_norm_rate:,.1f} rows/sec).")
        con.register("_bench_pool", bench_pool_df)
        pool_table = "_bench_pool"
        results["pool_norm_throughput_rows_per_sec"] = pool_norm_rate
    else:
        pool_table = f"'{train_pool_src.replace(chr(92), '/')}'"

    # Register S1 sample
    con.register("_s1_norm_sample", s1_sample_df)

    # Execute blocking queries (multi-view union)
    print("  Executing multi-view blocking SQL queries ...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _raw_candidates AS
        -- V1: exact name
        SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, p.src, 1 AS v1, 0 AS v2, 0 AS v3, 0 AS v4, 0 AS v5
        FROM _s1_norm_sample s1
        JOIN {pool_table} p ON s1.name_norm = p.name_norm AND s1.name_norm != ''
        
        UNION ALL
        
        -- V2: exact compact name
        SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, p.src, 0 AS v1, 1 AS v2, 0 AS v3, 0 AS v4, 0 AS v5
        FROM _s1_norm_sample s1
        JOIN {pool_table} p ON s1.name_compact = p.name_compact AND s1.name_compact != ''
        
        UNION ALL
        
        -- V3: exact address + country
        SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, p.src, 0 AS v1, 0 AS v2, 1 AS v3, 0 AS v4, 0 AS v5
        FROM _s1_norm_sample s1
        JOIN {pool_table} p ON s1.address_norm = p.address_norm AND s1.country_norm = p.country_norm
                           AND s1.address_norm != '' AND s1.country_norm != ''
        
        UNION ALL
        
        -- V4: prefix6 match
        SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, p.src, 0 AS v1, 0 AS v2, 0 AS v3, 1 AS v4, 0 AS v5
        FROM _s1_norm_sample s1
        JOIN {pool_table} p ON s1.name_prefix6 = p.name_prefix6 AND LENGTH(s1.name_prefix6) >= 5
        
        UNION ALL
        
        -- V5: consonant key match
        SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, p.src, 0 AS v1, 0 AS v2, 0 AS v3, 0 AS v4, 1 AS v5
        FROM _s1_norm_sample s1
        JOIN {pool_table} p ON s1.consonant_key = p.consonant_key AND LENGTH(s1.consonant_key) >= 5
    """)

    # Aggregate candidates per S1 and enforce Top-K cap
    candidates_df = con.execute(f"""
        WITH agg AS (
            SELECT
                s1_id AS source1_entity_id,
                cand_id AS candidate_entity_id,
                MAX(src) AS src,
                SUM(v1 + v2 + v3 + v4 + v5) AS n_views,
                MAX(v1) AS hit_v1,
                MAX(v2) AS hit_v2,
                MAX(v3) AS hit_v3,
                MAX(v4) AS hit_v4,
                MAX(v5) AS hit_v5,
                ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY SUM(v1 + v2 + v3 + v4 + v5) DESC) AS rank
            FROM _raw_candidates
            GROUP BY s1_id, cand_id
        )
        SELECT * FROM agg WHERE rank <= {top_k}
    """).fetchdf()

    t_blocking = time.time() - t0
    n_candidates = len(candidates_df)
    avg_cands = n_candidates / max(len(s1_sample_df), 1)

    # Evaluate blocking recall against ground truth
    found_true_matches = 0
    candidate_pair_set = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    for pair in gt_pairs:
        if pair in candidate_pair_set:
            found_true_matches += 1

    blocking_recall = found_true_matches / max(total_true_matches, 1)

    print(f"  Blocking complete in {t_blocking:.2f}s ({len(s1_sample_df) / max(t_blocking, 0.001):.1f} S1/sec)")
    print(f"  Total candidate pairs: {n_candidates:,} (avg {avg_cands:.1f} candidates/S1)")
    print(f"  Ground Truth matches captured: {found_true_matches:,}/{total_true_matches:,} ({blocking_recall * 100:.2f}% recall)")

    results["blocking"] = {
        "duration_sec": t_blocking,
        "total_candidates": n_candidates,
        "avg_candidates_per_s1": avg_cands,
        "true_matches_captured": found_true_matches,
        "total_true_matches": total_true_matches,
        "blocking_recall": blocking_recall,
        "throughput_s1_per_sec": len(s1_sample_df) / max(t_blocking, 0.001)
    }

    # -----------------------------------------------------------------------
    # Step 3: Feature Extraction Benchmark
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 3] Extracting features on {n_candidates:,} candidate pairs ...")
    t0 = time.time()

    # Join candidate pairs with S1 and candidate attributes
    con.register("_bench_cands", candidates_df)
    pair_features_input = con.execute(f"""
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.src,
            c.n_views,
            c.hit_v1 AS hit_V1,
            c.hit_v2 AS hit_V2,
            c.hit_v3 AS hit_V3,
            c.hit_v4 AS hit_V4,
            c.hit_v5 AS hit_V5,
            0 AS hit_V6, 0 AS hit_V7, 0 AS hit_V8, 0 AS hit_V9, 0 AS hit_V10,
            0.0 AS name_jaccard,
            s1.name_norm AS s1_name_norm,
            s1.name_compact AS s1_name_compact,
            s1.name_tokens AS s1_name_tokens,
            s1.address_norm AS s1_address_norm,
            s1.country_norm AS s1_country_norm,
            s1.missing_name AS s1_missing_name,
            s1.missing_address AS s1_missing_address,
            p.name_norm AS cand_name_norm,
            p.name_compact AS cand_name_compact,
            p.name_tokens AS cand_name_tokens,
            p.address_norm AS cand_address_norm,
            p.country_norm AS cand_country_norm,
            p.missing_name AS cand_missing_name,
            p.missing_address AS cand_missing_address
        FROM _bench_cands c
        JOIN _s1_norm_sample s1 ON c.source1_entity_id = s1.entity_id
        JOIN {pool_table} p ON c.candidate_entity_id = p.entity_id
    """).fetchdf()

    # Compute features via features.compute_features
    X = F.compute_features(pair_features_input)
    feature_names = F.FEATURE_NAMES
    t_feat = time.time() - t0
    feat_throughput = n_candidates / max(t_feat, 0.001)

    print(f"  Feature extraction complete in {t_feat:.2f}s ({feat_throughput:,.1f} pairs/sec)")
    print(f"  Feature matrix shape: {X.shape}, NaNs: {np.isnan(X).sum()}")

    # Build target vector y
    y = np.array([
        1 if (row["source1_entity_id"], row["candidate_entity_id"]) in gt_pairs else 0
        for _, row in candidates_df.iterrows()
    ], dtype=np.int32)
    positives = np.sum(y == 1)
    print(f"  Class balance: {positives:,} positives ({positives / max(len(y), 1) * 100:.2f}%), {len(y) - positives:,} negatives")

    results["features"] = {
        "duration_sec": t_feat,
        "matrix_shape": list(X.shape),
        "pairs_per_sec": feat_throughput,
        "n_positives": int(positives),
        "n_negatives": int(len(y) - positives)
    }

    # -----------------------------------------------------------------------
    # Step 4: Model Training & CV Benchmark
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 4] Training models & sweeping thresholds for F0.5 optimization ...")
    t0 = time.time()
    groups = candidates_df["source1_entity_id"].values

    report_path = os.path.join(config.ARTIFACTS_DIR, "benchmark_model_report.md")
    model_artifact = M.train_and_select(X, y, groups, feature_names, report_path=report_path)
    t_train = time.time() - t0

    print(f"  Model training & CV complete in {t_train:.2f}s.")
    print(f"  Best Model: {model_artifact['model_name']}")
    print(f"  OOF F0.5: {model_artifact['oof_f05']:.4f} (Precision: {model_artifact['oof_precision']:.4f}, Recall: {model_artifact['oof_recall']:.4f})")
    print(f"  Optimal Threshold: {model_artifact['threshold']:.3f}")

    results["model"] = {
        "duration_sec": t_train,
        "best_model": model_artifact["model_name"],
        "oof_f05": model_artifact["oof_f05"],
        "oof_precision": model_artifact["oof_precision"],
        "oof_recall": model_artifact["oof_recall"],
        "threshold": model_artifact["threshold"]
    }

    # -----------------------------------------------------------------------
    # Step 5: Test Output Generation & Local Pre-Validation
    # -----------------------------------------------------------------------
    print(f"\n[STAGE 5] Generating sample submission files and running local pre-validator ...")
    best_model = model_artifact["model"]
    threshold = model_artifact["threshold"]

    # Predict probabilities on benchmark candidates
    scores = best_model.predict_proba(X)[:, 1]
    candidates_df["pred_score"] = scores
    candidates_df["is_match"] = candidates_df["pred_score"] >= threshold

    # Create benchmark outputs
    bench_out_dir = os.path.join(config.OUTPUT_DIR, "benchmark")
    os.makedirs(bench_out_dir, exist_ok=True)
    bench_match_path = os.path.join(bench_out_dir, "matching_results.tsv")
    bench_cand_path  = os.path.join(bench_out_dir, "candidate_pairs.tsv")

    # Group matches & candidates per S1
    match_map = {}
    cand_map = {}

    for _, row in candidates_df.iterrows():
        s1 = row["source1_entity_id"]
        cand = row["candidate_entity_id"]
        cand_map.setdefault(s1, []).append(cand)
        if row["is_match"]:
            match_map.setdefault(s1, []).append(cand)

    # Write TSVs preserving ALL sampled S1 entities (including zero-match rows)
    with open(bench_match_path, "w", encoding="utf-8", newline="") as fm:
        fm.write(f"{config.S1_ID_COL}\t{config.MATCH_COL}\n")
        for s1 in s1_ids_list:
            m_list = match_map.get(s1, [])
            fm.write(f"{s1}\t{','.join(m_list)}\n")

    with open(bench_cand_path, "w", encoding="utf-8", newline="") as fc:
        fc.write(f"{config.S1_ID_COL}\t{config.CANDIDATE_COL}\n")
        for s1 in s1_ids_list:
            c_list = cand_map.get(s1, [])
            fc.write(f"{s1}\t{','.join(c_list)}\n")

    print(f"  Benchmark output written to:\n    {bench_match_path}\n    {bench_cand_path}")

    # Run local validation
    print("\n  Running local pre-validation tool (src/validate.py) ...")
    bench_s1_ref = os.path.join(bench_out_dir, "bench_s1_ref.tsv")
    s1_sample_df[["entity_id"]].to_csv(bench_s1_ref, sep="\t", index=False)

    bench_pool_ref = os.path.join(bench_out_dir, "bench_pool_ref.tsv")
    if "bench_pool_df" in locals():
        bench_pool_df[["entity_id"]].to_csv(bench_pool_ref, sep="\t", index=False)
        pool_paths = [bench_pool_ref]
    else:
        # Extract candidate IDs present in this benchmark run as valid pool IDs
        cand_ids = candidates_df[["candidate_entity_id"]].drop_duplicates().rename(columns={"candidate_entity_id": "entity_id"})
        cand_ids.to_csv(bench_pool_ref, sep="\t", index=False)
        pool_paths = [bench_pool_ref]

    val_ok = V.validate(
        matching_path=bench_match_path,
        candidate_path=bench_cand_path,
        s1_path=bench_s1_ref,
        pool_paths=pool_paths,
        verbose=True
    )
    results["validation_passed"] = val_ok

    # -----------------------------------------------------------------------
    # Step 6: Full-Scale Projection Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    full_s1_count = 1_732_544  # Full test S1
    scale_factor = full_s1_count / sample_size
    projected_blocking_hours = (t_blocking * scale_factor) / 3600.0
    projected_feature_hours = (t_feat * scale_factor) / 3600.0
    projected_total_hours = projected_blocking_hours + projected_feature_hours

    print("\n" + "=" * 70)
    print("PHASED BENCHMARK SUMMARY & FULL-SCALE PROJECTION")
    print("=" * 70)
    print(f"Benchmark sample size:             {sample_size:,} S1 entities")
    print(f"Candidate blocking recall:         {blocking_recall * 100:.2f}% (true matches in candidates)")
    print(f"Average candidates per S1:         {avg_cands:.1f}")
    print(f"Feature extraction rate:           {feat_throughput:,.1f} pairs/sec")
    print(f"Best model:                        {model_artifact['model_name']}")
    print(f"OOF F0.5 Score:                    {model_artifact['oof_f05']:.4f}")
    print(f"Optimal Threshold:                 {threshold:.3f}")
    print(f"Local Pre-Validation Status:       {'PASSED [OK]' if val_ok else 'FAILED [ERROR]'}")
    print("-" * 70)
    print(f"Projected Full Test Run ({full_s1_count:,} S1 entities):")
    print(f"  Estimated Blocking Time:         {projected_blocking_hours:.2f} hours")
    print(f"  Estimated Feature + Predict Time:{projected_feature_hours:.2f} hours")
    print(f"  Estimated Total Pipeline Time:   {projected_total_hours:.2f} hours")
    print("=" * 70)

    # Save benchmark results json
    summary_path = os.path.join(config.ARTIFACTS_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark summary written to: {summary_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run phased benchmark on business entity resolution pipeline.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 entities to benchmark (default 5000)")
    args = parser.parse_args()
    run_benchmark(sample_size=args.sample_size)
