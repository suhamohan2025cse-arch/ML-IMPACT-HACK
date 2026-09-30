"""
convert_to_parquet.py — STEP C: Convert all TSV datasets to Parquet.

Actions:
  1. Convert each TSV to Parquet using DuckDB (streaming, no full RAM load).
  2. Add all normalization columns via pandas apply (chunked for large files).
  3. Write final Parquet files with normalized columns.
  4. Verify row counts match raw TSV counts.
  5. Build combined candidate pools: train_pool and test_pool (S2+S3).

Usage:
  python src/convert_to_parquet.py

Run time estimate: ~5-10 minutes for all 7 files on this hardware.
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
import normalize as N

config.ensure_dirs()

# ---------------------------------------------------------------------------
# DuckDB connection helper
# ---------------------------------------------------------------------------
def get_con():
    con = duckdb.connect()
    for k, v in config.DUCKDB_CONFIG.items():
        try:
            con.execute(f"SET {k} = '{v}'")
        except Exception:
            pass   # older duckdb versions may not support all settings
    return con


# ---------------------------------------------------------------------------
# Step 1: Fast TSV -> raw Parquet via DuckDB (no Python row loops)
# ---------------------------------------------------------------------------
CONVERSIONS = [
    # (tsv_path, parquet_path, expected_rows)
    (config.TRAIN_S1_TSV, config.TRAIN_S1_PARQUET, 2_206_821),
    (config.TRAIN_S2_TSV, config.TRAIN_S2_PARQUET, 5_034_616),
    (config.TRAIN_S3_TSV, config.TRAIN_S3_PARQUET, 5_285_603),
    (config.TRAIN_GT_TSV, config.TRAIN_GT_PARQUET,  2_206_821),
    (config.TEST_S1_TSV,  config.TEST_S1_PARQUET,  1_732_544),
    (config.TEST_S2_TSV,  config.TEST_S2_PARQUET,  4_887_273),
    (config.TEST_S3_TSV,  config.TEST_S3_PARQUET,  5_082_316),
]


def tsv_to_raw_parquet(tsv_path: str, parquet_path: str, expected_rows: int):
    """
    Use DuckDB to stream TSV -> Parquet.
    All columns are read as VARCHAR to avoid type-inference surprises.
    """
    if os.path.exists(parquet_path):
        print(f"  [SKIP] {os.path.basename(parquet_path)} already exists.")
        return

    t0 = time.time()
    con = get_con()
    # Read with all columns as VARCHAR; write to Parquet
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv(
                '{tsv_path.replace(chr(92), '/')}',
                delim='\\t',
                header=true,
                all_varchar=true,
                ignore_errors=false
            )
        ) TO '{parquet_path.replace(chr(92), '/')}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    con.close()

    # Verify row count
    con2 = get_con()
    count = con2.execute(
        f"SELECT COUNT(*) FROM '{parquet_path.replace(chr(92), '/')}'"
    ).fetchone()[0]
    con2.close()

    elapsed = time.time() - t0
    status = "OK" if count == expected_rows else f"MISMATCH (got {count}, expected {expected_rows})"
    print(f"  {os.path.basename(parquet_path)}: {count:,} rows in {elapsed:.1f}s — {status}")
    if count != expected_rows:
        raise ValueError(f"Row count mismatch for {parquet_path}")


# ---------------------------------------------------------------------------
# Step 2: Add normalized columns (chunked pandas apply + pyarrow write)
# ---------------------------------------------------------------------------
SOURCE_FILES = [
    # (raw_parquet, norm_parquet_path, is_ground_truth)
    (config.TRAIN_S1_PARQUET, config.TRAIN_S1_PARQUET, False),
    (config.TRAIN_S2_PARQUET, config.TRAIN_S2_PARQUET, False),
    (config.TRAIN_S3_PARQUET, config.TRAIN_S3_PARQUET, False),
    (config.TEST_S1_PARQUET,  config.TEST_S1_PARQUET,  False),
    (config.TEST_S2_PARQUET,  config.TEST_S2_PARQUET,  False),
    (config.TEST_S3_PARQUET,  config.TEST_S3_PARQUET,  False),
]

NORM_SENTINEL = "_norm_done_"


def needs_normalization(parquet_path: str) -> bool:
    """Return True if the Parquet file does not yet have normalized columns."""
    if not os.path.exists(parquet_path):
        return True
    pf = pq.ParquetFile(parquet_path)
    return "name_norm" not in pf.schema_arrow.names


def add_norm_columns(parquet_path: str, chunk_size: int = 200_000):
    """
    Read Parquet in chunks, add normalization columns, rewrite in place.
    Avoids loading the full ~5M row file into RAM at once.
    """
    if not needs_normalization(parquet_path):
        print(f"  [SKIP] {os.path.basename(parquet_path)} already normalized.")
        return

    t0 = time.time()
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    tmp_path = parquet_path + ".tmp"

    writer = None
    rows_done = 0

    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        # Normalize
        df["name_norm"]    = N.apply_name_norm(df["business_name"])
        df["name_compact"] = N.apply_name_compact(df["name_norm"])
        df["address_norm"] = N.apply_address_norm(df["business_address"])
        df["country_norm"] = N.apply_country_norm(df["country"])

        # Token strings (comma-separated for easy DuckDB string_split)
        df["name_tokens"]    = N.apply_name_tokens_str(df["name_norm"])
        df["address_tokens"] = N.apply_address_tokens_str(df["address_norm"])

        # Derived blocking helpers
        df["consonant_key"]  = N.apply_consonant_skeleton(df["name_compact"])
        df["house_number"]   = N.apply_house_number(df["address_norm"])

        # Missingness flags (integer 0/1 for DuckDB aggregation)
        df["missing_name"]    = (df["business_name"].fillna("").str.strip() == "").astype("int8")
        df["missing_address"] = (df["business_address"].fillna("").str.strip() == "").astype("int8")

        # Prefix helpers for blocking (prefix4 of name tokens)
        # Stored as strings; rare1/2/3 computed later using token_df
        df["name_prefix6"] = df["name_compact"].str[:6]
        df["name_prefix8_last8"] = df["name_compact"].str[:8] + "|" + df["name_compact"].str[-8:]

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
        writer.write_table(table)

        rows_done += len(df)
        pct = 100 * rows_done / max(total_rows, 1)
        print(f"    {os.path.basename(parquet_path)}: {rows_done:,}/{total_rows:,} ({pct:.0f}%)", end="\r")

    if writer:
        writer.close()

    # Replace original — use retry loop to handle OneDrive/antivirus file locks on Windows
    import shutil, time as _time
    for attempt in range(5):
        try:
            os.replace(tmp_path, parquet_path)
            break
        except PermissionError:
            if attempt < 4:
                print(f"\n    [WARN] File lock on {os.path.basename(parquet_path)}, retrying in 3s (attempt {attempt+1}/5)...")
                _time.sleep(3)
            else:
                # Final fallback: copy then delete
                shutil.copy2(tmp_path, parquet_path)
                os.remove(tmp_path)

    elapsed = time.time() - t0
    print(f"\n  [DONE] {os.path.basename(parquet_path)} normalized in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Step 3: Build combined candidate pools (S2 + S3)
# ---------------------------------------------------------------------------
def build_pool(s2_parquet: str, s3_parquet: str, pool_parquet: str, tag: str):
    """
    Union S2 and S3 into a single candidate pool Parquet.
    Adds a 'src' column ('S2' or 'S3') so features can flag source origin.
    """
    if os.path.exists(pool_parquet):
        print(f"  [SKIP] {os.path.basename(pool_parquet)} already exists.")
        return

    t0 = time.time()
    con = get_con()
    s2 = s2_parquet.replace("\\", "/")
    s3 = s3_parquet.replace("\\", "/")
    out = pool_parquet.replace("\\", "/")

    con.execute(f"""
        COPY (
            SELECT *, 'S2' AS src FROM '{s2}'
            UNION ALL
            SELECT *, 'S3' AS src FROM '{s3}'
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    count = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
    con.close()
    elapsed = time.time() - t0
    print(f"  {os.path.basename(pool_parquet)}: {count:,} rows in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE C — TSV to Parquet Conversion")
    print("=" * 60)

    # Step 1: Raw TSV -> Parquet
    print("\n[1/3] Converting TSVs to raw Parquet ...")
    for tsv, parquet, expected in CONVERSIONS:
        print(f"  Processing {os.path.basename(tsv)} ...")
        tsv_to_raw_parquet(tsv, parquet, expected)

    # Step 2: Add normalization columns to source files
    print("\n[2/3] Adding normalization columns ...")
    for raw_pq, norm_pq, is_gt in SOURCE_FILES:
        print(f"  Normalizing {os.path.basename(raw_pq)} ...")
        add_norm_columns(norm_pq)

    # Step 3: Build combined pools
    print("\n[3/3] Building candidate pools ...")
    build_pool(
        config.TRAIN_S2_PARQUET, config.TRAIN_S3_PARQUET,
        config.TRAIN_POOL_PARQUET, "train"
    )
    build_pool(
        config.TEST_S2_PARQUET, config.TEST_S3_PARQUET,
        config.TEST_POOL_PARQUET, "test"
    )

    print("\n[DONE] Parquet conversion complete.")
    print(f"Files in: {config.PARQUET_DIR}")
    for f in os.listdir(config.PARQUET_DIR):
        path = os.path.join(config.PARQUET_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")
