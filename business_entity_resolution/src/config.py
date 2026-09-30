"""
config.py — Central configuration for the Business Entity Resolution pipeline.
All paths and hyperparameters live here so every module imports from one place.
"""
import os

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = r"C:\Users\mohan\Downloads\Amazon_ML_Challenge_2026"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

# Raw TSVs
TRAIN_S1_TSV = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_TSV = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_TSV = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_TSV = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_S1_TSV  = os.path.join(TEST_DIR,  "test_source1.tsv")
TEST_S2_TSV  = os.path.join(TEST_DIR,  "test_source2.tsv")
TEST_S3_TSV  = os.path.join(TEST_DIR,  "test_source3.tsv")

# ---------------------------------------------------------------------------
# Parquet paths
# ---------------------------------------------------------------------------
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")

TRAIN_S1_PARQUET = os.path.join(PARQUET_DIR, "train_source1.parquet")
TRAIN_S2_PARQUET = os.path.join(PARQUET_DIR, "train_source2.parquet")
TRAIN_S3_PARQUET = os.path.join(PARQUET_DIR, "train_source3.parquet")
TRAIN_GT_PARQUET = os.path.join(PARQUET_DIR, "train_ground_truth.parquet")
TEST_S1_PARQUET  = os.path.join(PARQUET_DIR, "test_source1.parquet")
TEST_S2_PARQUET  = os.path.join(PARQUET_DIR, "test_source2.parquet")
TEST_S3_PARQUET  = os.path.join(PARQUET_DIR, "test_source3.parquet")

# Combined candidate pool parquets (S2+S3 with src tag)
TRAIN_POOL_PARQUET     = os.path.join(PARQUET_DIR, "train_pool.parquet")
TEST_POOL_PARQUET      = os.path.join(PARQUET_DIR, "test_pool.parquet")

# Token document-frequency tables
TRAIN_TOKEN_DF_PARQUET = os.path.join(PARQUET_DIR, "train_token_df.parquet")
TEST_TOKEN_DF_PARQUET  = os.path.join(PARQUET_DIR, "test_token_df.parquet")

# ---------------------------------------------------------------------------
# Intermediate / artifact paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR        = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH           = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURE_COLS_PATH    = os.path.join(ARTIFACTS_DIR, "feature_cols.json")
BLOCKING_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "blocking_report.md")
ABLATION_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "ablation.md")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR           = os.path.join(PROJECT_ROOT, "output")
MATCHING_RESULTS_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_TSV  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
DEBUG_DIR            = os.path.join(OUTPUT_DIR, "debug")
EXPLANATIONS_TSV     = os.path.join(DEBUG_DIR, "explanations.tsv")

# Intermediate Parquets during prediction
PRED_CANDIDATES_PARQUET = os.path.join(PARQUET_DIR, "pred_candidates.parquet")
PRED_MATCHES_PARQUET    = os.path.join(PARQUET_DIR, "pred_matches.parquet")

# ---------------------------------------------------------------------------
# DuckDB settings — tuned for 16 GB total / ~1.65 GB free RAM
# DuckDB spills aggressively to disk; 243 GB free disk is ample.
# ---------------------------------------------------------------------------
TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")

DUCKDB_CONFIG = {
    "memory_limit": "8GB",
    "temp_directory": TMP_DIR,
    "threads": "4",
}

# ---------------------------------------------------------------------------
# Blocking hyperparameters
# ---------------------------------------------------------------------------
# Drop blocking keys mapping to more than MAX_BLOCK_SIZE records (avoids giant blocks)
MAX_BLOCK_SIZE = 50
# Max candidates retained per S1 entity after multi-view union + ranking
TOP_K = 50
# Stop-token detection: tokens appearing in > this fraction of pool = stop token
STOP_TOKEN_FREQ_THRESHOLD = 0.01   # data-driven, no hardcoded word lists

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
PREDICT_CHUNK_SIZE = 100_000
TRAIN_SAMPLE_SIZE  = 30_000

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
N_CV_FOLDS           = 5
THRESHOLD_SWEEP_MIN  = 0.05
THRESHOLD_SWEEP_MAX  = 0.95
THRESHOLD_SWEEP_STEP = 0.01
RANDOM_STATE         = 42

# ---------------------------------------------------------------------------
# Output format constants (confirmed from Phase A inspection)
# ---------------------------------------------------------------------------
TSV_SEP       = "\t"
ID_LIST_SEP   = ","
MATCH_COL     = "matched_entity_ids"
CANDIDATE_COL = "candidate_entity_ids"
S1_ID_COL     = "source1_entity_id"

# ---------------------------------------------------------------------------
# Ensure all required directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in [PARQUET_DIR, ARTIFACTS_DIR, OUTPUT_DIR, DEBUG_DIR, TMP_DIR]:
        os.makedirs(d, exist_ok=True)
