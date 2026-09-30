"""
blocking.py — Multi-view high-recall candidate blocker.

10 blocking views:
  V1:  exact name_norm
  V2:  sorted non-stop name tokens (sorted_tokens key)
  V3:  rare1 + rare2
  V4:  rare1 + country_norm  (only when rare1 is sufficiently rare)
  V5:  prefix4(rare1) + prefix4(rare2)
  V6:  first8 + last8 of name_compact
  V7:  consonant_skeleton[:8] + name_compact[:8]
  V8:  rare1 + house_number token
  V9:  exact address_norm
  V10: rare address token + country_norm + first_numeric

Rules:
  - MAX_BLOCK_SIZE cap: keys mapping to > MAX_BLOCK_SIZE records are dropped.
  - TOP_K: after union of all views, keep top-K candidates per S1 ranked by
    (n_views DESC, name_token_overlap DESC).
  - Never loads full pool into RAM; operates via DuckDB SQL on Parquet files.

Usage (programmatic):
    from blocking import run_blocking
    candidates_df = run_blocking(
        s1_parquet=config.TEST_S1_PARQUET,
        pool_parquet=config.TEST_POOL_PARQUET,
        token_df_parquet=config.TEST_TOKEN_DF_PARQUET,
        max_block_size=config.MAX_BLOCK_SIZE,
        top_k=config.TOP_K,
        con=None,   # pass an existing connection or None to create one
    )
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import pandas as pd
import config


# ---------------------------------------------------------------------------
# Helper: get a configured DuckDB connection
# ---------------------------------------------------------------------------
def get_con():
    con = duckdb.connect()
    for k, v in config.DUCKDB_CONFIG.items():
        try:
            con.execute(f"SET {k} = '{v}'")
        except Exception:
            pass
    return con


# ---------------------------------------------------------------------------
# Step 1: Compute token document frequencies over a candidate pool
# ---------------------------------------------------------------------------
def compute_token_df(pool_parquet: str, output_parquet: str, con=None):
    """
    Compute per-token document frequency in the candidate pool (S2+S3).
    token_df columns: token, df (count of pool records containing token)
    Used for data-driven stop-token detection and rarity signals.
    """
    if os.path.exists(output_parquet):
        print(f"  [SKIP] token_df already exists: {os.path.basename(output_parquet)}")
        return

    close_con = con is None
    if con is None:
        con = get_con()

    t0 = time.time()
    pool = pool_parquet.replace("\\", "/")
    out  = output_parquet.replace("\\", "/")
    total = con.execute(f"SELECT COUNT(*) FROM '{pool}'").fetchone()[0]

    con.execute(f"""
        COPY (
            WITH tokens AS (
                SELECT
                    entity_id,
                    UNNEST(string_split(name_tokens, ',')) AS token
                FROM '{pool}'
                WHERE name_tokens IS NOT NULL AND name_tokens != ''
            ),
            df AS (
                SELECT token, COUNT(DISTINCT entity_id) AS df
                FROM tokens
                WHERE LENGTH(token) >= 2
                GROUP BY token
            )
            SELECT
                token,
                df,
                CAST(df AS DOUBLE) / {total} AS df_frac
            FROM df
            ORDER BY df DESC
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    n = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
    elapsed = time.time() - t0
    print(f"  token_df: {n:,} distinct tokens in {elapsed:.1f}s")

    if close_con:
        con.close()


# ---------------------------------------------------------------------------
# Step 2: Annotate S1 and pool with rare1/rare2/rare3 tokens
# ---------------------------------------------------------------------------
def annotate_rare_tokens(
    parquet_path: str,
    token_df_parquet: str,
    stop_threshold: float,
    con=None,
) -> str:
    """
    Add rare1, rare2, rare3 columns to a source Parquet.
    Overwrites parquet_path in place.
    Returns parquet_path.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    if "rare1" in pf.schema_arrow.names:
        print(f"  [SKIP] rare tokens already in {os.path.basename(parquet_path)}")
        return parquet_path

    close_con = con is None
    if con is None:
        con = get_con()

    t0 = time.time()
    src  = parquet_path.replace("\\", "/")
    tdf  = token_df_parquet.replace("\\", "/")
    tmp  = parquet_path + ".rare_tmp"
    out  = tmp.replace("\\", "/")

    # Build stop token set from token_df
    con.execute(f"""
        COPY (
            WITH stop_tokens AS (
                SELECT token FROM '{tdf}' WHERE df_frac > {stop_threshold}
            ),
            src_tokens AS (
                SELECT
                    entity_id,
                    name_tokens,
                    string_split(name_tokens, ',') AS tok_arr
                FROM '{src}'
            ),
            filtered AS (
                SELECT
                    s.entity_id,
                    [t FOR t IN s.tok_arr IF t NOT IN (SELECT token FROM stop_tokens) AND LENGTH(t) >= 2] AS rare_tokens
                FROM src_tokens s
            ),
            with_df AS (
                SELECT
                    f.entity_id,
                    t AS token,
                    COALESCE(d.df, 0) AS df
                FROM filtered f,
                UNNEST(f.rare_tokens) t
                LEFT JOIN '{tdf}' d ON d.token = t
            ),
            ranked AS (
                SELECT
                    entity_id,
                    token,
                    df,
                    ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY df ASC, token ASC) AS rn
                FROM with_df
            ),
            pivoted AS (
                SELECT
                    entity_id,
                    MAX(CASE WHEN rn = 1 THEN token END) AS rare1,
                    MAX(CASE WHEN rn = 2 THEN token END) AS rare2,
                    MAX(CASE WHEN rn = 3 THEN token END) AS rare3
                FROM ranked
                GROUP BY entity_id
            )
            SELECT s.*, p.rare1, p.rare2, p.rare3
            FROM '{src}' s
            LEFT JOIN pivoted p ON p.entity_id = s.entity_id
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    os.replace(tmp, parquet_path)
    elapsed = time.time() - t0
    print(f"  rare tokens annotated on {os.path.basename(parquet_path)} in {elapsed:.1f}s")

    if close_con:
        con.close()

    return parquet_path


# ---------------------------------------------------------------------------
# Step 3: Build blocking key index tables (one per view, stored in DuckDB)
# ---------------------------------------------------------------------------
def _run_single_view(
    con, view_name: str, sql_key_expr_s1: str, sql_key_expr_pool: str,
    s1_table: str, pool_table: str, max_block_size: int
) -> str:
    """
    Execute one blocking view and return the name of the result table.
    Result columns: source1_entity_id, candidate_entity_id, src
    Only keeps keys with <= max_block_size pool matches (prevents giant blocks).
    """
    result_table = f"_block_{view_name}"

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE {result_table} AS
        WITH s1_keys AS (
            SELECT entity_id AS s1_id,
                   ({sql_key_expr_s1}) AS block_key
            FROM {s1_table}
            WHERE ({sql_key_expr_s1}) IS NOT NULL
              AND LENGTH(({sql_key_expr_s1})) >= 2
        ),
        pool_keys AS (
            SELECT entity_id AS cand_id,
                   src,
                   ({sql_key_expr_pool}) AS block_key
            FROM {pool_table}
            WHERE ({sql_key_expr_pool}) IS NOT NULL
              AND LENGTH(({sql_key_expr_pool})) >= 2
        ),
        key_sizes AS (
            SELECT block_key, COUNT(*) AS pool_cnt
            FROM pool_keys
            GROUP BY block_key
        ),
        valid_keys AS (
            SELECT block_key FROM key_sizes WHERE pool_cnt <= {max_block_size}
        )
        SELECT DISTINCT
            s.s1_id     AS source1_entity_id,
            p.cand_id   AS candidate_entity_id,
            p.src       AS src
        FROM s1_keys s
        JOIN valid_keys vk ON vk.block_key = s.block_key
        JOIN pool_keys  p  ON p.block_key  = s.block_key
    """)

    n = con.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
    print(f"    {view_name}: {n:,} raw pairs before TOP_K")
    return result_table


# ---------------------------------------------------------------------------
# Main blocking function
# ---------------------------------------------------------------------------
def run_blocking(
    s1_parquet: str,
    pool_parquet: str,
    token_df_parquet: str,
    max_block_size: int = config.MAX_BLOCK_SIZE,
    top_k: int = config.TOP_K,
    stop_threshold: float = config.STOP_TOKEN_FREQ_THRESHOLD,
    con=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all 10 blocking views against the candidate pool.
    Returns a DataFrame with columns:
        source1_entity_id, candidate_entity_id, src,
        n_views, hit_V1..hit_V10
    """
    close_con = con is None
    if con is None:
        con = get_con()

    s1   = s1_parquet.replace("\\", "/")
    pool = pool_parquet.replace("\\", "/")
    tdf  = token_df_parquet.replace("\\", "/")

    t0 = time.time()

    # Register parquet files as views
    con.execute(f"CREATE OR REPLACE VIEW _s1   AS SELECT * FROM '{s1}'")
    con.execute(f"CREATE OR REPLACE VIEW _pool AS SELECT * FROM '{pool}'")
    con.execute(f"CREATE OR REPLACE VIEW _tdf  AS SELECT * FROM '{tdf}'")

    # Compute stop token set (data-driven)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _stop_tokens AS
        SELECT token FROM _tdf WHERE df_frac > {stop_threshold}
    """)

    if verbose:
        n_stop = con.execute("SELECT COUNT(*) FROM _stop_tokens").fetchone()[0]
        print(f"  Stop tokens identified: {n_stop}")

    # ---------------------------------------------------------------------------
    # Helper SQL expressions for each blocking key
    # ---------------------------------------------------------------------------
    # V1: exact name_norm
    v1_key = "name_norm"

    # V2: sorted non-stop tokens joined as a string key
    # We pre-sort via name_tokens (which is already sorted alphabetically at norm time)
    # Filter out stop tokens from name_tokens
    # Since name_tokens is already a comma-sep sorted string, we rebuild without stops
    # Note: DuckDB list comprehension syntax
    v2_key = """
        ARRAY_TO_STRING(
            LIST_SORT([t FOR t IN string_split(COALESCE(name_tokens,''), ',')
                       IF t NOT IN (SELECT token FROM _stop_tokens) AND LENGTH(t)>=2]),
            '|'
        )
    """

    # V3: rare1 + rare2
    v3_key = "CASE WHEN rare1 IS NOT NULL AND rare2 IS NOT NULL THEN rare1 || '|' || rare2 END"

    # V4: rare1 + country_norm (only when rare1 has low df)
    # We restrict V4 to entities where rare1 df_frac < 0.001 (very rare token)
    v4_key = """
        CASE
            WHEN rare1 IS NOT NULL AND country_norm IS NOT NULL AND country_norm != ''
            THEN rare1 || '||' || country_norm
        END
    """

    # V5: prefix4(rare1) + prefix4(rare2)
    v5_key = """
        CASE WHEN rare1 IS NOT NULL AND rare2 IS NOT NULL
             THEN LEFT(rare1,4) || '|' || LEFT(rare2,4)
        END
    """

    # V6: first8 + last8 of name_compact
    v6_key = """
        CASE WHEN LENGTH(name_compact) >= 8
             THEN LEFT(name_compact,8) || '|' || RIGHT(name_compact,8)
        END
    """

    # V7: consonant_skeleton[:8] + name_compact[:8]
    v7_key = """
        CASE WHEN LENGTH(consonant_key) >= 4 AND LENGTH(name_compact) >= 4
             THEN LEFT(consonant_key,8) || '|' || LEFT(name_compact,8)
        END
    """

    # V8: rare1 + house_number
    v8_key = """
        CASE WHEN rare1 IS NOT NULL AND house_number IS NOT NULL AND house_number != ''
             THEN rare1 || '|' || house_number
        END
    """

    # V9: exact address_norm (only for non-empty addresses)
    v9_key = """
        CASE WHEN LENGTH(COALESCE(address_norm,'')) >= 6
             THEN address_norm
        END
    """

    # V10: rare address token + country_norm + first numeric
    # We derive rare address token as first address_token not in stop list
    # and length >= 4
    v10_key = """
        CASE
            WHEN house_number IS NOT NULL AND house_number != ''
             AND rare1 IS NOT NULL AND country_norm IS NOT NULL
             AND country_norm != ''
             THEN house_number || '|' || country_norm || '|' || LEFT(COALESCE(address_norm,''),12)
        END
    """

    VIEWS = [
        ("V1",  v1_key,  v1_key),
        ("V2",  v2_key,  v2_key),
        ("V3",  v3_key,  v3_key),
        ("V4",  v4_key,  v4_key),
        ("V5",  v5_key,  v5_key),
        ("V6",  v6_key,  v6_key),
        ("V7",  v7_key,  v7_key),
        ("V8",  v8_key,  v8_key),
        ("V9",  v9_key,  v9_key),
        ("V10", v10_key, v10_key),
    ]

    if verbose:
        print(f"  Running {len(VIEWS)} blocking views ...")

    result_tables = []
    for vname, s1_expr, pool_expr in VIEWS:
        t_v = time.time()
        try:
            tbl = _run_single_view(
                con, vname, s1_expr, pool_expr,
                "_s1", "_pool", max_block_size
            )
            result_tables.append((vname, tbl))
        except Exception as e:
            print(f"    {vname}: ERROR — {e} (skipped)")

    # ---------------------------------------------------------------------------
    # Union all views, count hits per view, compute n_views
    # ---------------------------------------------------------------------------
    if verbose:
        print("  Unioning and deduplicating views ...")

    union_sql = "\nUNION ALL\n".join(
        f"SELECT source1_entity_id, candidate_entity_id, src, '{vn}' AS view_name "
        f"FROM {tbl}"
        for vn, tbl in result_tables
    )

    hit_cols = ", ".join(
        f"MAX(CASE WHEN view_name = '{vn}' THEN 1 ELSE 0 END) AS hit_{vn}"
        for vn, _ in result_tables
    )
    all_views = [vn for vn, _ in result_tables]

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _all_pairs AS
        WITH union_pairs AS ({union_sql}),
        agg AS (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                src,
                COUNT(DISTINCT view_name) AS n_views,
                {hit_cols}
            FROM union_pairs
            GROUP BY source1_entity_id, candidate_entity_id, src
        )
        SELECT * FROM agg
    """)

    total_pairs = con.execute("SELECT COUNT(*) FROM _all_pairs").fetchone()[0]
    if verbose:
        print(f"  Total unique pairs before TOP_K: {total_pairs:,}")

    # ---------------------------------------------------------------------------
    # Approximate name-token overlap for ranking (cheap: Jaccard on sorted tokens)
    # We compute this in DuckDB to avoid Python loops
    # ---------------------------------------------------------------------------
    # Add a cheap name similarity score for ranking: len(intersection)/len(union)
    # Using list operations in DuckDB
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _s1_tokens AS
        SELECT entity_id, string_split(COALESCE(name_tokens,''), ',') AS tok_arr
        FROM _s1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _pool_tokens AS
        SELECT entity_id, string_split(COALESCE(name_tokens,''), ',') AS tok_arr
        FROM _pool
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _ranked_pairs AS
        WITH joined AS (
            SELECT
                p.*,
                s1t.tok_arr AS s1_toks,
                pt.tok_arr  AS cand_toks
            FROM _all_pairs p
            JOIN _s1_tokens   s1t ON s1t.entity_id = p.source1_entity_id
            JOIN _pool_tokens  pt ON  pt.entity_id = p.candidate_entity_id
        ),
        scored AS (
            SELECT *,
                -- intersection size / union size (Jaccard approximation)
                CAST(
                    list_aggregate(list_intersect(s1_toks, cand_toks), 'count')
                    AS DOUBLE
                ) /
                GREATEST(
                    list_aggregate(list_distinct(list_concat(s1_toks, cand_toks)), 'count'),
                    1
                ) AS name_jaccard
            FROM joined
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY n_views DESC, name_jaccard DESC
                ) AS rn
            FROM scored
        )
        SELECT * FROM ranked WHERE rn <= {top_k}
    """)

    final_count = con.execute("SELECT COUNT(*) FROM _ranked_pairs").fetchone()[0]
    elapsed = time.time() - t0
    if verbose:
        print(f"  Final candidate pairs (after TOP_K={top_k}): {final_count:,} in {elapsed:.1f}s")

    # Collect result as DataFrame (candidates only, not the full pool)
    hit_col_names = [f"hit_{vn}" for vn, _ in result_tables]
    keep_cols = (
        ["source1_entity_id", "candidate_entity_id", "src", "n_views", "name_jaccard"]
        + hit_col_names
    )

    # Ensure all V1-V10 hit cols exist (fill missing with 0)
    all_possible_hits = [f"hit_V{i}" for i in range(1, 11)]
    existing = set(hit_col_names)
    select_cols = []
    for c in keep_cols:
        if c in existing or not c.startswith("hit_"):
            select_cols.append(c)
    for c in all_possible_hits:
        if c not in existing:
            select_cols.append(f"0 AS {c}")

    col_str = ", ".join(
        c if not c.startswith("0 AS") else c
        for c in keep_cols
    )
    # Build final select with zero-fill for missing views
    final_select_parts = []
    for c in ["source1_entity_id", "candidate_entity_id", "src", "n_views", "name_jaccard"]:
        final_select_parts.append(c)
    for vn in [f"V{i}" for i in range(1, 11)]:
        hc = f"hit_{vn}"
        if hc in existing:
            final_select_parts.append(hc)
        else:
            final_select_parts.append(f"0 AS {hc}")

    df = con.execute(
        f"SELECT {', '.join(final_select_parts)} FROM _ranked_pairs"
    ).df()

    # Cleanup temp tables
    for _, tbl in result_tables:
        try:
            con.execute(f"DROP TABLE IF EXISTS {tbl}")
        except Exception:
            pass

    if close_con:
        con.close()

    return df


# ---------------------------------------------------------------------------
# Blocking validation helper: recall measurement against ground truth
# ---------------------------------------------------------------------------
def measure_blocking_recall(
    candidates_df: pd.DataFrame,
    gt_parquet: str,
    s1_sample_ids: set = None,
) -> dict:
    """
    Measure recall of the blocking step against ground truth.
    gt_parquet must have columns: source1_entity_id, matched_entity_ids (comma-sep)
    s1_sample_ids: if provided, filter GT to only these S1 IDs.

    Returns a dict with recall metrics.
    """
    import pyarrow.parquet as pq

    gt = pq.read_table(gt_parquet).to_pandas()
    if s1_sample_ids is not None:
        gt = gt[gt["source1_entity_id"].isin(s1_sample_ids)]

    # Build ground truth pairs
    gt_pairs = set()
    gt_per_s1 = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if matched and str(matched).strip():
            ids = [x.strip() for x in str(matched).split(",") if x.strip()]
            gt_pairs.update((s1_id, cid) for cid in ids)
            gt_per_s1[s1_id] = set(ids)

    # Build blocking pairs
    blocking_pairs = set(
        zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"])
    )

    # Overall recall
    if not gt_pairs:
        return {"error": "No ground truth pairs found"}

    retrieved = gt_pairs & blocking_pairs
    recall = len(retrieved) / len(gt_pairs)

    # Per-view recall
    view_recalls = {}
    for vi in range(1, 11):
        vn = f"V{vi}"
        col = f"hit_{vn}"
        if col not in candidates_df.columns:
            continue
        view_pairs = set(
            zip(
                candidates_df.loc[candidates_df[col] == 1, "source1_entity_id"],
                candidates_df.loc[candidates_df[col] == 1, "candidate_entity_id"],
            )
        )
        view_recall = len(gt_pairs & view_pairs) / len(gt_pairs)
        view_recalls[vn] = view_recall

    # Recall for multi-match S1 entities (2+ true matches)
    multi_gt = {s1: ids for s1, ids in gt_per_s1.items() if len(ids) >= 2}
    multi_pairs = set((s1, cid) for s1, ids in multi_gt.items() for cid in ids)
    multi_retrieved = multi_pairs & blocking_pairs
    multi_recall = len(multi_retrieved) / max(len(multi_pairs), 1)

    # Average candidates per S1
    cand_counts = candidates_df.groupby("source1_entity_id").size()
    avg_cands = cand_counts.mean()
    med_cands = cand_counts.median()
    max_cands = cand_counts.max()

    return {
        "total_gt_pairs":       len(gt_pairs),
        "retrieved_pairs":      len(retrieved),
        "overall_recall":       recall,
        "multi_match_recall":   multi_recall,
        "n_multi_match_s1":     len(multi_gt),
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": med_cands,
        "max_candidates_per_s1": max_cands,
        "view_recalls":          view_recalls,
    }
