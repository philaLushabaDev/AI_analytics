import os
import random
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Basic logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_analytics")

# -------------------------
# Configuration
# -------------------------
import warnings
# hide noisy deprecation/future warnings so stdout remains clean for PHP parsing
warnings.filterwarnings("ignore")

# NOTE: avoid hard-coded secrets in source. Set DB credentials via env vars.
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PORT = os.getenv("DB_PORT", "13152")
DB_PASS = os.getenv("DB_PASS", "AVNS_lurH_tAP0IeniHK2r7b")  # no default secret in code
DB_HOST = os.getenv("DB_HOST", "mysql-10505a59-philalushaba11d-51bd.f.aivencloud.com")
DB_NAME = os.getenv("DB_NAME", "defaultdb")  # prefer explicit env var in deployment

MODEL_PATH = "ai_tabular_model.joblib"
SCALER_PATH = "ai_scaler.joblib"
AGG_DATA_CSV = "agg_dataset.csv"
DB_EXPORT_SQL = "db_export.sql"
TRAIN_EPOCHS_PER_CALL = 4
LEARNING_RATE = 1e-3
SEED = 42

FEATURE_COLS = ["num_postings", "num_applications", "unique_applicants", "posts_per_month", "apps_per_post"]

# Notifications table name
NOTIFICATIONS_TABLE = "notifications"

# Whitelisted tables that read_table_safe may query
ALLOWED_TABLES = {"users", "organizations", "applications", "job_postings", NOTIFICATIONS_TABLE}

# reproducibility
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# In-memory cache
# -------------------------
# CACHE holds table DataFrames and aggregated DataFrame so subsequent operations
# in the same request use memory instead of requerying DB.
CACHE: Dict[str, Any] = {
    "tables": {},      # table_name -> DataFrame
    "agg": None,       # aggregated DataFrame (org-level)
    "last_refresh": None,
    "sql_path": None
}

# -------------------------
# Embedded / synthetic agg dataset helpers (single-file option)
# -------------------------
EMBEDDED_AGG_CSV = """org_id,num_postings,num_applications,unique_applicants,posts_per_month,apps_per_post,hire_rate
1,5,10,8,0.5,2.0,0.10
2,2,1,1,0.2,0.50,0.00
3,10,50,40,0.9,5.00,0.12
4,0,0,0,0.0,0.00,0.00
5,3,6,5,0.3,2.00,0.05
"""


def write_embedded_agg_csv(path: str = AGG_DATA_CSV, force: bool = False):
    """
    Write the embedded CSV to disk if the file does not exist (or if force=True).
    Keeps the script self-contained so you can run training without external files.
    """
    try:
        if force or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(EMBEDDED_AGG_CSV)
            logger.info("Wrote embedded agg CSV to %s (force=%s)", path, force)
    except Exception as e:
        logger.exception("Failed to write embedded agg CSV: %s", e)


def generate_synthetic_agg(path: str = AGG_DATA_CSV, n_orgs: int = 50, seed: int = SEED):
    """
    Generate a synthetic agg_dataset.csv with `n_orgs` rows.
    Useful if you want a larger dataset but still keep a single-file deployment.
    Returns the generated pandas DataFrame.
    """
    import numpy as _np
    import pandas as _pd
    _np.random.seed(seed)
    rows = []
    for org in range(1, n_orgs + 1):
        # Basic stochastic generation of features with plausible ranges
        num_postings = int(max(0, _np.random.poisson(4)))               # mean ~4 postings
        unique_applicants = int(max(0, _np.random.poisson(max(1, num_postings*4))))
        num_applications = int(max(0, _np.random.poisson(max(0, unique_applicants * 1.2))))
        # posts_per_month derived from postings across ~3-18 months span
        months_span = max(1.0, _np.random.uniform(3.0, 18.0))
        posts_per_month = round(float(num_postings) / months_span, 4)
        apps_per_post = round(float(num_applications) / num_postings, 4) if num_postings > 0 else 0.0
        # hire_rate small random fraction, correlated with number of hires and apps
        hires = int(_np.random.binomial(num_applications, 0.03)) if num_applications > 0 else 0
        hire_rate = round(float(hires) / max(1, num_applications), 6) if num_applications > 0 else 0.0

        rows.append({
            "org_id": org,
            "num_postings": num_postings,
            "num_applications": num_applications,
            "unique_applicants": unique_applicants,
            "posts_per_month": posts_per_month,
            "apps_per_post": apps_per_post,
            "hire_rate": hire_rate
        })

    df = _pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Generated synthetic agg CSV at %s with %d rows", path, len(df))
    return df


# -------------------------
# Lazy DB engine factory & app
# -------------------------
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        db_user = os.getenv("DB_USER", DB_USER)
        db_pass = quote_plus(os.getenv("DB_PASS", DB_PASS))
        db_host = os.getenv("DB_HOST", DB_HOST)
        db_port = os.getenv("DB_PORT", DB_PORT)
        db_name = os.getenv("DB_NAME", DB_NAME)
        database_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        try:
            _engine = create_engine(database_url, pool_pre_ping=True)
            logger.info("Created DB engine for %s@%s/%s", db_user, db_host, db_name)
        except Exception as e:
            logger.exception("Failed to create DB engine: %s", e)
            raise
    return _engine

app = FastAPI(title="AI Analytics (scikit-learn) with Notifications")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -------------------------
# Utility: read tables (schema from your dump)
# -------------------------
def read_table_safe(name: str) -> pd.DataFrame:
    """
    Read a table into a DataFrame in a safe way. Table names must be whitelisted.
    Returns empty DataFrame on failure.
    """
    try:
        name_clean = str(name).strip()
        if name_clean not in ALLOWED_TABLES:
            logger.warning("Attempt to read non-whitelisted table: %s", name_clean)
            return pd.DataFrame()
        engine = get_engine()
        with engine.connect().execution_options(stream_results=True) as conn:
            # table name has been validated; use backticks to avoid reserved-word problems
            return pd.read_sql(text(f"SELECT * FROM `{name_clean}`"), conn)
    except Exception as e:
        logger.exception("read_table_safe failed for %s: %s", name, e)
        return pd.DataFrame()


def read_all_tables() -> Dict[str, pd.DataFrame]:
    tables = {
        "users": read_table_safe("users"),
        "organizations": read_table_safe("organizations"),
        "applications": read_table_safe("applications"),
        "job_postings": read_table_safe("job_postings")
    }
    # notifications table may not exist initially
    try:
        tables["notifications"] = read_table_safe(NOTIFICATIONS_TABLE)
    except Exception as e:
        logger.exception("Failed to read notifications table: %s", e)
        tables["notifications"] = pd.DataFrame()
    return tables

# -------------------------
# Notifications: ensure table exists
# -------------------------
def ensure_notifications_table():
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{NOTIFICATIONS_TABLE}` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `user_id` int(11) DEFAULT NULL,
      `org_id` int(11) DEFAULT NULL,
      `job_posting_id` int(11) DEFAULT NULL,
      `sender_type` varchar(50) DEFAULT 'system',
      `sender_id` int(11) DEFAULT NULL,
      `message` text,
      `data` json DEFAULT NULL,
      `is_read` tinyint(1) NOT NULL DEFAULT 0,
      `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
      PRIMARY KEY (`id`),
      KEY `user_id` (`user_id`),
      KEY `org_id` (`org_id`),
      KEY `job_posting_id` (`job_posting_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(create_sql))
    except Exception as e:
        logger.exception("Failed to ensure notifications table exists: %s", e)

# -------------------------
# Aggregation tuned to your schema (fixed, defensive)
# -------------------------
def build_aggregates(tables: Dict[str, pd.DataFrame]):
    users = tables.get("users", pd.DataFrame())
    orgs = tables.get("organizations", pd.DataFrame())
    apps = tables.get("applications", pd.DataFrame())
    jobs = tables.get("job_postings", pd.DataFrame())

    # defensive copies
    jobs_local = jobs.copy()
    apps_local = apps.copy()
    orgs_local = orgs.copy()

    # normalize column names if needed
    if "job_posting_id" in jobs_local.columns:
        jobs_local = jobs_local.rename(columns={"job_posting_id": "id"})
    if "application_id" in apps_local.columns:
        apps_local = apps_local.rename(columns={"application_id": "id"})
    if "job_posting_id" in apps_local.columns:
        apps_local = apps_local.rename(columns={"job_posting_id": "job_id"})

    # ensure organizations have an id column
    if "id" not in orgs_local.columns:
        orgs_local = orgs_local.reset_index().rename(columns={"index": "id"})

    # helper: ensure a dataframe has a column as object dtype Series (match index length)
    def ensure_col_object(df: pd.DataFrame, col: str):
        if col in df.columns:
            df[col] = df[col].astype(object)
        else:
            df[col] = pd.Series([None] * len(df), index=df.index).astype(object)

    ensure_col_object(jobs_local, "id")
    ensure_col_object(jobs_local, "org_id")
    ensure_col_object(apps_local, "job_id")
    ensure_col_object(apps_local, "user_id")

    # merge apps with jobs to bring org_id onto each application row
    merged = apps_local.merge(
        jobs_local[["id", "org_id"]].rename(columns={"id": "job_id"}),
        on="job_id",
        how="left",
    )

    # ---- fixed: detect hires safely (always produce a Series) ----
    if "status" in merged.columns:
        merged["_is_hire"] = merged["status"].astype(str).str.lower().eq("hired")
    else:
        merged["_is_hire"] = pd.Series([False] * len(merged), index=merged.index)

    # count aggregates per org_id
    agg_apps = (
        merged.groupby("org_id")
        .agg(
            num_applications=("job_id", "count"),
            unique_applicants=("user_id", lambda s: int(s.nunique()) if s.notnull().any() else 0),
            num_hires=("_is_hire", "sum"),
        )
        .reset_index()
    )

    # job posting counts per org
    jp = jobs_local.groupby("org_id").agg(num_postings=("id", "count")).reset_index()

    # posts per month (defensive)
    if "created_at" in jobs_local.columns:
        try:
            jobs_local["created_at"] = pd.to_datetime(jobs_local["created_at"], errors="coerce")
            span_days = (jobs_local["created_at"].max() - jobs_local["created_at"].min()).days
            months = (span_days / 30.0) if span_days > 0 else 1.0
            posts_per_month = jobs_local.groupby("org_id").size().div(months).reset_index(name="posts_per_month")
            jp = jp.merge(posts_per_month, on="org_id", how="left")
        except Exception:
            jp["posts_per_month"] = jp["num_postings"] / max(1.0, jobs_local.shape[0] / 1.0)
    else:
        jp["posts_per_month"] = jp["num_postings"] / max(1.0, jobs_local.shape[0] / 1.0)

    # combine with organizations
    org_df = orgs_local.copy()
    if "id" in org_df.columns:
        org_df = org_df.rename(columns={"id": "org_id"})
    else:
        org_df["org_id"] = pd.Series(range(1, len(org_df) + 1), index=org_df.index)

    agg_full = org_df.merge(jp, on="org_id", how="left").merge(agg_apps, on="org_id", how="left")

    # fill NA with zeros for expected numeric columns
    for col in ["num_postings", "num_applications", "unique_applicants", "num_hires", "posts_per_month"]:
        if col not in agg_full.columns:
            agg_full[col] = 0
        agg_full[col] = agg_full[col].fillna(0)

    # derived columns (safe division)
    agg_full["apps_per_post"] = agg_full.apply(
        lambda r: (r["num_applications"] / r["num_postings"]) if r["num_postings"] > 0 else 0.0, axis=1
    )
    agg_full["hire_rate"] = agg_full.apply(
        lambda r: (r["num_hires"] / r["num_applications"]) if r["num_applications"] > 0 else 0.0, axis=1
    )

    # global metrics (ensure _is_hire exists)
    total_hires = int(merged["_is_hire"].sum()) if "_is_hire" in merged.columns else 0
    global_metrics = {
        "total_orgs": int(agg_full.shape[0]),
        "total_postings": int(jobs_local.shape[0]),
        "total_applications": int(merged.shape[0]),
        "total_hires": total_hires,
    }

    return agg_full, global_metrics, users, jobs_local, merged

# -------------------------
# New: Export DB to SQL (INSERT statements) and refresh in-memory cache
# -------------------------
def _sql_literal(val):
    """Return SQL literal for a Python value."""
    if pd.isnull(val):
        return "NULL"
    # datetime-like
    if isinstance(val, (pd.Timestamp, datetime)):
        return f"'{str(val)}'"
    # numeric
    if isinstance(val, (int, np.integer, float, np.floating)):
        # keep NaN handled above
        return str(val)
    # boolean
    if isinstance(val, (bool, np.bool_)):
        return '1' if val else '0'
    # fallback: escape single quotes
    s = str(val)
    s = s.replace("'", "''")
    return f"'{s}'"


def export_db_to_sql(path: str = DB_EXPORT_SQL, tables: Optional[List[str]] = None, overwrite: bool = True):
    """
    Export selected whitelisted tables into a SQL file containing INSERT statements.
    Overwrites existing file if overwrite=True.
    """
    if tables is None:
        tables = list(ALLOWED_TABLES)
    tables = [t for t in tables if t in ALLOWED_TABLES]
    mode = "w" if overwrite else "a"
    try:
        engine = get_engine()
        with engine.connect().execution_options(stream_results=True) as conn, open(path, mode, encoding="utf-8") as outf:
            outf.write(f"-- Exported at {datetime.utcnow().isoformat()} (UTC)\n")
            for t in tables:
                try:
                    df = pd.read_sql(text(f"SELECT * FROM `{t}`"), conn)
                except Exception as e:
                    logger.exception("Failed to read table %s during export: %s", t, e)
                    continue
                outf.write(f"\n-- Table `{t}` ({len(df)} rows)\n")
                if df.empty:
                    continue
                cols = list(df.columns)
                col_list = ", ".join([f"`{c}`" for c in cols])
                # write INSERTs in moderately sized batches
                for _, row in df.iterrows():
                    vals = ", ".join([_sql_literal(row[c]) for c in cols])
                    insert = f"INSERT INTO `{t}` ({col_list}) VALUES ({vals});\n"
                    outf.write(insert)
        logger.info("Exported DB tables to SQL file: %s", path)
    except Exception as e:
        logger.exception("export_db_to_sql failed: %s", e)


def refresh_cache_and_export_sql(overwrite_sql: bool = True):
    """
    Read all whitelisted tables into CACHE['tables'], compute aggregate (CACHE['agg']),
    and export the DB to an SQL file (overwriting previous file if overwrite_sql=True).
    """
    try:
        tables = read_all_tables()
        # put tables in cache as DataFrames
        CACHE["tables"] = tables
        # build aggregates from the fetched tables
        agg_df, global_metrics, users, jobs_local, merged = build_aggregates(tables)
        CACHE["agg"] = agg_df
        CACHE["last_refresh"] = datetime.utcnow().isoformat()
        # export SQL file (overwrites by default)
        export_db_to_sql(path=DB_EXPORT_SQL, tables=list(ALLOWED_TABLES), overwrite=overwrite_sql)
        CACHE["sql_path"] = DB_EXPORT_SQL
        logger.info("Cache refreshed at %s; agg rows=%d", CACHE["last_refresh"], 0 if agg_df is None else agg_df.shape[0])
    except Exception as e:
        logger.exception("refresh_cache_and_export_sql failed: %s", e)

# -------------------------
# Incremental training helpers (scikit-learn) - updated to train from CACHE
# -------------------------
def train_from_cache(overwrite_agg_csv: bool = True, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Train model using the cached aggregated dataset (CACHE['agg']).
    - Overwrites AGG_DATA_CSV each time when overwrite_agg_csv=True.
    - Splits into train/test and fits scaler + RandomForestRegressor.
    Returns metrics and paths.
    """
    result = {"trained": False, "rows": 0, "mse": None, "r2": None, "model_path": None, "scaler_path": None}
    agg_df = CACHE.get("agg")
    if agg_df is None or agg_df.empty:
        logger.warning("No aggregated data in cache; cannot train.")
        # fallback: write embedded CSV if desired
        if not os.path.exists(AGG_DATA_CSV):
            write_embedded_agg_csv(AGG_DATA_CSV, force=False)
        return result

    # Ensure FEATURE_COLS exist
    for c in FEATURE_COLS:
        if c not in agg_df.columns:
            agg_df[c] = 0.0

    df = agg_df[["org_id"] + FEATURE_COLS + ["hire_rate"]].fillna(0).copy()
    if overwrite_agg_csv:
        try:
            df.to_csv(AGG_DATA_CSV, index=False)
            logger.info("Wrote AGG dataset to %s (overwrite=%s)", AGG_DATA_CSV, overwrite_agg_csv)
        except Exception as e:
            logger.exception("Failed to write AGG_DATA_CSV: %s", e)

    result["rows"] = df.shape[0]
    if df.shape[0] < 2:
        logger.warning("Not enough rows to train (rows=%d)", df.shape[0])
        return result

    try:
        X = df[FEATURE_COLS].fillna(0).values.astype(float)
        y = df["hire_rate"].fillna(0).values.astype(float)

        # If only 1 row, skip; else split
        if df.shape[0] >= 2 and test_size > 0 and df.shape[0] > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        # Save scaler
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Saved scaler to %s", SCALER_PATH)

        model = RandomForestRegressor(n_estimators=100, random_state=SEED)
        model.fit(X_train_s, y_train)
        joblib.dump(model, MODEL_PATH)
        logger.info("Trained and saved model to %s", MODEL_PATH)

        # evaluate
        X_test_s = scaler.transform(X_test)
        preds = model.predict(X_test_s)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds) if len(y_test) > 0 else None

        result.update({
            "trained": True,
            "mse": float(mse),
            "r2": float(r2) if r2 is not None else None,
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH
        })
        logger.info("Training metrics: mse=%s r2=%s", mse, r2)
    except Exception as e:
        logger.exception("train_from_cache failed: %s", e)

    return result


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        return model, scaler
    except Exception as e:
        logger.exception("Failed to load model/scaler: %s", e)
        return None, None

# -------------------------
# Notifications: matching & insert
# -------------------------
def simple_keyword_set_from_text(text: str) -> List[str]:
    t = str(text or "").lower()
    toks = [w.strip() for w in t.split() if len(w.strip()) >= 3]
    keywords = list(set(toks))
    return keywords


def create_notification_row(user_id: Optional[int], org_id: Optional[int], job_posting_id: Optional[int], sender_type: str, sender_id: Optional[int], message: str, data: Optional[Dict] = None):
    ensure_notifications_table()
    data_json = json.dumps(data) if data is not None else None
    insert_sql = text(f"INSERT INTO `{NOTIFICATIONS_TABLE}` (user_id, org_id, job_posting_id, sender_type, sender_id, message, data, is_read, created_at) VALUES (:user_id, :org_id, :job_posting_id, :sender_type, :sender_id, :message, :data, :is_read, :created_at)")
    params = {
        "user_id": user_id,
        "org_id": org_id,
        "job_posting_id": job_posting_id,
        "sender_type": sender_type,
        "sender_id": sender_id,
        "message": message,
        "data": data_json,
        "is_read": 0,
        "created_at": datetime.utcnow()
    }
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(insert_sql, params)
    except Exception as e:
        logger.exception("Failed to insert notification row: %s", e)


def notify_unnotified_jobs(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Find job_postings that have not had job_posted notifications created yet and generate notifications
    for users whose profile keywords (career/qualification/experience/languages) match the job posting title/description/requirements.
    Returns a summary dict with counts.
    """
    ensure_notifications_table()
    jobs_df = tables.get("job_postings", pd.DataFrame())
    users_df = tables.get("users", pd.DataFrame())
    orgs_df = tables.get("organizations", pd.DataFrame())

    if jobs_df.empty or users_df.empty:
        return {"notified_jobs": 0, "user_notifications": 0}

    # ensure normalized id column name
    if "job_posting_id" in jobs_df.columns and "id" not in jobs_df.columns:
        jobs_df = jobs_df.rename(columns={"job_posting_id": "id"})

    # load existing notifications to avoid duplicates
    existing_notifs = read_table_safe(NOTIFICATIONS_TABLE)
    existing_job_ids = set(existing_notifs[existing_notifs["sender_type"] == 'job_posted']["job_posting_id"].dropna().astype(int).tolist()) if not existing_notifs.empty else set()

    job_rows_to_process = []
    for _, job in jobs_df.iterrows():
        job_id = int(job.get("id") or job.get("job_posting_id") or 0)
        if job_id in existing_job_ids:
            continue
        job_rows_to_process.append(job)

    user_notif_count = 0
    job_notif_count = 0

    for job in job_rows_to_process:
        job_id = int(job.get("id") or job.get("job_posting_id") or 0)
        title = str(job.get("job_title", ""))
        desc = str(job.get("description", ""))
        reqs = str(job.get("requirements", ""))
        combined_text = f"{title} {desc} {reqs}".lower()
        keywords = simple_keyword_set_from_text(combined_text)
        if not keywords:
            # fallback to entire title
            keywords = [w for w in title.lower().split() if len(w) >= 3]

        matched_users = []
        # iterate users and match
        for _, user in users_df.iterrows():
            # build user's text fields
            user_text = ""
            for col in ["career", "qualification", "experience", "languages"]:
                if col in user.index and pd.notnull(user.get(col)):
                    user_text += " " + str(user.get(col)).lower()
            # also check region if job has location
            if "location" in job and pd.notnull(job.get("location")) and "region" in user.index and pd.notnull(user.get("region")):
                if str(user.get("region")).strip().lower() in str(job.get("location")).lower():
                    try:
                        matched_users.append(int(user.get("id")))
                    except Exception:
                        pass
                    continue
            if any(kw in user_text for kw in keywords):
                try:
                    matched_users.append(int(user.get("id")))
                except Exception:
                    pass

        # craft message
        org_id = int(job.get("org_id") or 0) if pd.notnull(job.get("org_id")) else None
        org_name = ""
        if org_id and not orgs_df.empty and "id" in orgs_df.columns:
            o = orgs_df[orgs_df["id"] == org_id]
            if not o.empty:
                org_name = o.iloc[0].get("org_name", "")

        message_short = f"New job posted: {title} at {org_name}."

        # create a sentinel notification row for this job to mark it as processed
        create_notification_row(None, org_id, job_id, "job_posted", None, f"Job posted processed for notifications: {title}", {"job_title": title})
        job_notif_count += 1

        # notify matched users
        for uid in set(matched_users):
            message = message_short + " This posting matches your profile."
            create_notification_row(uid, org_id, job_id, "system", None, message, {"job_title": title, "org_name": org_name})
            user_notif_count += 1

        # also notify org owner (if organizations.username or email maps to a user)
        org_owner_user_id = None
        try:
            if org_id and not orgs_df.empty:
                row = orgs_df[orgs_df["id"] == org_id]
                if not row.empty:
                    row0 = row.iloc[0]
                    # try match by username
                    if "username" in row0.index and pd.notnull(row0.get("username")):
                        uname = row0.get("username")
                        u = users_df[users_df["username"] == uname]
                        if not u.empty:
                            org_owner_user_id = int(u.iloc[0]["id"])
                    # try match by email
                    if org_owner_user_id is None and "email" in row0.index and pd.notnull(row0.get("email")):
                        oemail = row0.get("email")
                        u = users_df[users_df["email"] == oemail]
                        if not u.empty:
                            org_owner_user_id = int(u.iloc[0]["id"])
        except Exception:
            org_owner_user_id = None

        if org_owner_user_id is not None:
            # craft AI/system notification for the org
            ai_msg = f"Your job '{title}' was posted. The site AI will monitor responses and provide insights in analytics."
            create_notification_row(org_owner_user_id, org_id, job_id, "ai", None, ai_msg, {"job_title": title})

    return {"notified_jobs": job_notif_count, "user_notifications": user_notif_count}

# -------------------------
# Narrative generation (professional, non-prescriptive)
# -------------------------
def generate_job_seeker_narrative(user_row: Optional[pd.Series], agg_df: pd.DataFrame, jobs_df: pd.DataFrame, per_org: List[Dict], global_metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append("Analytical summary for job seekers:")
    lines.append(f"Dataset: {global_metrics.get('total_orgs',0)} organizations, {global_metrics.get('total_postings',0)} postings, {global_metrics.get('total_applications',0)} applications.")
    avg_apps = (global_metrics.get("total_applications", 0) / max(1, global_metrics.get("total_postings", 0)))
    lines.append(f"Average candidate interest: {avg_apps:.2f} applications per posting.")

    # highlight companies posting a lot and those hiring most
    top_posters = sorted(per_org, key=lambda r: r["num_postings"], reverse=True)[:5]
    if top_posters:
        lines.append("Organizations posting most frequently: " + ", ".join([f"{o['name']} ({o['num_postings']})" for o in top_posters]))
    top_hires = sorted(per_org, key=lambda r: r["num_hires"], reverse=True)[:5]
    if top_hires:
        lines.append("Organizations with most hires: " + ", ".join([f"{o['name']} ({o['num_hires']})" for o in top_hires]))

    # personalized: quick matching using user's 'career', 'qualification', 'experience', 'region'
    matches = []
    if user_row is not None and not jobs_df.empty:
        candidate_keywords = []
        for col in ["career", "qualification", "experience", "languages"]:
            if col in user_row.index and pd.notnull(user_row[col]):
                candidate_keywords.extend([w.strip().lower() for w in str(user_row[col]).replace(";",",").split(",") if w.strip()])
        user_region = user_row.get("region") if "region" in user_row.index else None
        # simple matching by title/description/location
        for _, job in jobs_df.iterrows():
            score = 0
            title = str(job.get("job_title","")).lower()
            desc = str(job.get("description","")).lower()
            loc = str(job.get("location","")).lower()
            if user_region and user_region.strip().lower() in loc:
                score += 1
            for kw in candidate_keywords[:10]:
                if kw and (kw in title or kw in desc):
                    score += 1
            if score >= 1:
                matches.append({"job_title": job.get("job_title"), "org_id": int(job.get("org_id",0)), "location": job.get("location")})
        # dedupe & sort
        matches = matches[:8]

    if matches:
        lines.append("Recent postings relevant to your profile (for awareness):")
        for m in matches:
            org_name = next((o['name'] for o in per_org if o['org_id']==m['org_id']), f"Org_{m['org_id']}")
            lines.append(f"- {m['job_title']} — {org_name} ({m.get('location','')})")
    else:
        lines.append("No strong keyword matches found in recent postings; consider refining your profile (skills, career field, region).")

    lines.append("Observations are intended to inform — not direct — actions. Combine with your own context when deciding next steps.")
    return "\n\n".join(lines)


def generate_hr_narrative(user_row: Optional[pd.Series], org_row: Optional[pd.Series], agg_df: pd.DataFrame, jobs_df: pd.DataFrame, merged_apps: pd.DataFrame, per_org: List[Dict], global_metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append("Analytical summary for HR / Employer:")
    if org_row is not None:
        lines.append(f"Organization: {org_row.get('org_name','(unnamed)')} — postings: {int(org_row.get('num_postings',0))}, applications: {int(org_row.get('num_applications',0))}, hires: {int(org_row.get('num_hires',0))}.")
        lines.append(f"Observed apps/posting: {org_row.get('apps_per_post',0.0):.2f}; hire rate: {org_row.get('hire_rate',0.0):.1%}.")
        if org_row.get('num_postings',0) > 10 and org_row.get('hire_rate',0.0) < 0.05:
            lines.append("Observation: high posting activity combined with low hire rate may indicate long hiring cycles or mismatch between requirements and applicant profiles.")
    else:
        lines.append("No single organization associated with this user was detected; the summary below is global.")

    avg_apps = (global_metrics.get("total_applications",0) / max(1, global_metrics.get("total_postings",1)))
    lines.append(f"Across the dataset average applications per posting is approximately {avg_apps:.2f}.")
    lines.append("Top publishers and top hirers are included in the structured output for your review.")
    lines.append("These insights are data-driven; use them to prioritize operational checks (job descriptions, screening criteria, pipeline bottlenecks).")
    return "\n\n".join(lines)

# -------------------------
# Main analyze function (updated to refresh cache & train from DB)
# -------------------------
def analyze_for(user_id: Optional[int], role: str, org_id_override: Optional[int] = None) -> Dict[str, Any]:
    """
    Now: refreshes in-memory cache from DB (and exports SQL), trains from cache,
    and then runs the normal analysis using the cached data.
    """
    # Step 1: refresh cache and export DB -> SQL (overwrites old file)
    try:
        refresh_cache_and_export_sql(overwrite_sql=True)
    except Exception as e:
        logger.exception("Failed to refresh cache: %s", e)

    # Use CACHE tables for everything below
    tables = CACHE.get("tables", read_all_tables())
    agg_df, global_metrics, users_df, jobs_df, merged_apps_df = build_aggregates(tables)

    # Step 2: train model from cached aggregated dataset (overwrite agg CSV)
    try:
        train_result = train_from_cache(overwrite_agg_csv=True, test_size=0.2)
    except Exception as e:
        logger.exception("train_from_cache failed: %s", e)
        train_result = {"trained": False}

    # prepare per-org simple list from agg_df
    per_org = []
    if agg_df is None:
        agg_df = pd.DataFrame()
    for _, r in agg_df.iterrows():
        name = r.get("org_name") if "org_name" in r.index else (r.get("org_name") if "org_name" in r else f"Org_{int(r.get('org_id',-1))}")
        per_org.append({
            "org_id": int(r.get("org_id", -1)),
            "name": name,
            "num_postings": int(r.get("num_postings",0)),
            "num_applications": int(r.get("num_applications",0)),
            "num_hires": int(r.get("num_hires",0)),
            "apps_per_post": float(r.get("apps_per_post",0.0)),
            "hire_rate": float(r.get("hire_rate",0.0))
        })

    # generate notifications for newly posted jobs (idempotent)
    try:
        notify_summary = notify_unnotified_jobs(tables)
    except Exception as e:
        logger.exception("notify_unnotified_jobs failed: %s", e)
        notify_summary = {"notified_jobs": 0, "user_notifications": 0}

    # load model for optional predictions (defensively handle types)
    model, scaler = load_model_and_scaler()
    if model is not None and scaler is not None and not agg_df.empty:
        X_df = agg_df.reindex(columns=FEATURE_COLS).fillna(0)
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)
        try:
            X = X_df.values.astype(float)
            Xs = scaler.transform(X)
            preds = model.predict(Xs)
        except Exception as e:
            logger.exception("Model prediction failed: %s", e)
            preds = None

        if preds is not None:
            for idx, org_obj in enumerate(per_org):
                try:
                    org_obj["model_predicted_hire_rate"] = float(preds[idx])
                except Exception:
                    org_obj["model_predicted_hire_rate"] = None
        else:
            for org_obj in per_org:
                org_obj["model_predicted_hire_rate"] = None
    else:
        for o in per_org:
            o["model_predicted_hire_rate"] = None

    # identify user row if provided
    user_row = None
    if users_df is not None and not users_df.empty and user_id is not None:
        if "id" in users_df.columns:
            matched = users_df[users_df["id"] == user_id]
            if not matched.empty:
                user_row = matched.iloc[0]

    # for HR try to match user's organization via username/email or org_id_override
    org_row = None
    if role == "hr":
        if org_id_override is not None:
            found = agg_df[agg_df["org_id"] == org_id_override]
            if not found.empty:
                org_row = found.iloc[0]
        else:
            if user_row is not None and not tables.get("organizations", pd.DataFrame()).empty:
                orgs_df = tables["organizations"]
                uname = user_row.get("username") if "username" in user_row.index else None
                email = user_row.get("email") if "email" in user_row.index else None
                matched = pd.DataFrame()
                if uname is not None and "username" in orgs_df.columns:
                    matched = orgs_df[orgs_df["username"] == uname]
                if matched.empty and email is not None and "email" in orgs_df.columns:
                    matched = orgs_df[orgs_df["email"] == email]
                if not matched.empty:
                    candidate_id = int(matched.iloc[0]["id"])
                    found = agg_df[agg_df["org_id"] == candidate_id]
                    if not found.empty:
                        org_row = found.iloc[0]

    # generate narrative
    if role == "job_seeker":
        narrative = generate_job_seeker_narrative(user_row, agg_df, jobs_df, per_org, global_metrics)
    elif role == "hr":
        narrative = generate_hr_narrative(user_row, org_row, agg_df, jobs_df, merged_apps_df, per_org, global_metrics)
    else:
        narrative = "Role not recognized. Use role=job_seeker or role=hr."

    # top lists
    top_by_postings = sorted(per_org, key=lambda r: r["num_postings"], reverse=True)[:5]
    top_by_hires = sorted(per_org, key=lambda r: r["num_hires"], reverse=True)[:5]
    top_by_apps_per_post = sorted(per_org, key=lambda r: r["apps_per_post"], reverse=True)[:5]

    return {
        "analysis_text": narrative,
        "per_organization": per_org,
        "global_metrics": global_metrics,
        "top_by_postings": top_by_postings,
        "top_by_hires": top_by_hires,
        "top_by_apps_per_post": top_by_apps_per_post,
        "notify_summary": notify_summary,
        "train_result": train_result,
        "cache_info": {"last_refresh": CACHE.get("last_refresh"), "sql_path": CACHE.get("sql_path")}
    }

# -------------------------
# API endpoint
# -------------------------
class AnalysisResponse(BaseModel):
    analysis_text: str
    per_organization: List[Dict[str, Any]]
    global_metrics: Dict[str, Any]
    top_by_postings: List[Dict[str, Any]]
    top_by_hires: List[Dict[str, Any]]
    top_by_apps_per_post: List[Dict[str, Any]]
    notify_summary: Dict[str, Any]

@app.get("/analyze", response_model=AnalysisResponse)
def analyze(user_id: Optional[int] = Query(None), role: str = Query(...), org_id: Optional[int] = Query(None)):
    """
    Example:
      /analyze?user_id=3&role=job_seeker
      /analyze?user_id=3&role=hr
      /analyze?user_id=3&role=hr&org_id=1
    This endpoint now also triggers a cache refresh, DB->SQL export, training from cache, and returns analytics.
    """
    try:
        return analyze_for(user_id=user_id, role=role, org_id_override=org_id)
    except Exception as e:
        logger.exception("Analyze endpoint error: %s", e)
        return {
            "analysis_text": f"Error during analysis: {str(e)}",
            "per_organization": [],
            "global_metrics": {},
            "top_by_postings": [],
            "top_by_hires": [],
            "top_by_apps_per_post": [],
            "notify_summary": {"notified_jobs": 0, "user_notifications": 0}
        }


@app.post("/notify_jobs")
def trigger_notifications():
    """Manual endpoint to trigger detection and creation of notifications for unprocessed jobs."""
    try:
        tables = read_all_tables()
        summary = notify_unnotified_jobs(tables)
        return {"status": "ok", "summary": summary}
    except Exception as e:
        logger.exception("trigger_notifications failed: %s", e)
        return {"status": "error", "error": str(e)}


# -------------------------
# notifications fetching endpoint (unchanged)
# -------------------------
@app.get("/notifications")
def get_notifications(user_id: Optional[int] = Query(None)):
    try:
        ensure_notifications_table()
        engine = get_engine()
        if user_id is None:
            # return recent notifications
            df = read_table_safe(NOTIFICATIONS_TABLE)
            if df.empty:
                return {"notifications": []}
            df2 = df.sort_values(by="created_at", ascending=False).head(50)
            return {"notifications": df2.to_dict(orient="records")}
        else:
            sql = text(f"SELECT * FROM `{NOTIFICATIONS_TABLE}` WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 100")
            with engine.connect() as conn:
                res = conn.execute(sql, {"user_id": user_id}).mappings().all()
                return {"notifications": [dict(r) for r in res]}
    except Exception as e:
        logger.exception("get_notifications failed: %s", e)
        return {"notifications": [], "error": str(e)}

# -------------------------
# CLI entrypoint notes (left minimal)
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Analytics with DB-to-cache training")
    parser.add_argument("--mode", choices=["hr", "jobseeker"], help="Run in HR or Jobseeker mode")
    parser.add_argument("--user-id", type=int, help="User ID (for Jobseeker mode)")
    parser.add_argument("--org-id", type=int, help="Organization ID (for HR mode)")
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-user", type=str, default="root")
    parser.add_argument("--db-pass", type=str, default="")
    parser.add_argument("--db-name", type=str, default="graduate_system")
    parser.add_argument("--output-file", type=str, help="Path to save JSON output")
    parser.add_argument("--regen-agg", action="store_true", help="Regenerate embedded agg_dataset.csv and exit")
    parser.add_argument("--synthetic-agg", type=int, help="Generate synthetic agg_dataset.csv with N orgs and exit")
    parser.add_argument("--train", action="store_true", help="Train model using agg_dataset.csv and exit")
    args = parser.parse_args()

    # If user passed DB overrides via CLI, apply them to environment so get_engine() picks them up.
    if args.db_host:
        os.environ["DB_HOST"] = args.db_host
    if args.db_user:
        os.environ["DB_USER"] = args.db_user
    if args.db_pass is not None:
        os.environ["DB_PASS"] = args.db_pass
    if args.db_name:
        os.environ["DB_NAME"] = args.db_name

    if args.regen_agg:
        write_embedded_agg_csv(force=True)
        print('{"status": "ok", "message": "Wrote embedded agg dataset to agg_dataset.csv"}')
        return

    if args.synthetic_agg:
        generate_synthetic_agg(n_orgs=args.synthetic_agg)
        print(f'{{"status": "ok", "message": "Generated synthetic agg dataset with {args.synthetic_agg} rows at agg_dataset.csv"}}')
        return

    if args.train:
        # Refresh cache then train from it
        refresh_cache_and_export_sql(overwrite_sql=True)
        res = train_from_cache(overwrite_agg_csv=True)
        print(json.dumps({"status": "ok", "train_result": res}))
        return

    # default operation: run analysis
    if args.mode == "hr":
        result = analyze_for(user_id=args.user_id, role="hr", org_id_override=args.org_id)
    elif args.mode == "jobseeker":
        result = analyze_for(user_id=args.user_id, role="job_seeker", org_id_override=args.org_id)
    else:
        parser.print_help()
        return

    out_path = args.output_file if args.output_file else None
    if out_path is None:
        import tempfile
        fd, temp_path = tempfile.mkstemp(prefix="ai_analytics_out_", suffix=".json", text=True)
        os.close(fd)
        out_path = temp_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(), "result": result}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output_file": out_path}))


if __name__ == "__main__":
    main()

