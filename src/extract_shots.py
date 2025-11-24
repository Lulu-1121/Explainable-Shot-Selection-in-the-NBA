# src/extract_shots.py
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DB, PROCESSED_DIR


def time_to_seconds(t):
    """Convert 'MM:SS' string to seconds. Return NaN if invalid."""
    if not isinstance(t, str):
        return np.nan
    try:
        parts = t.split(":")
        if len(parts) != 2:
            return np.nan
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except Exception:
        return np.nan


def parse_score_margin(x):
    """Convert scoremargin string to integer. 'TIE' -> 0, invalid -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    if isinstance(x, (int, float)):
        return int(x)

    s = str(x).strip().upper()
    if s == "" or s == "TIE":
        return 0
    try:
        s = s.replace("+", "")
        return int(s)
    except Exception:
        return np.nan


def infer_home_offense(home_desc, visitor_desc):
    """Infer whether the offense is home team from description text."""
    home_has = isinstance(home_desc, str) and home_desc.strip() != ""
    vis_has = isinstance(visitor_desc, str) and visitor_desc.strip() != ""

    if home_has and not vis_has:
        return 1  # home on offense
    if vis_has and not home_has:
        return 0  # visitor on offense
    # Ambiguous or both empty
    return np.nan


def detect_three_point(home_desc, visitor_desc):
    """Detect if the shot is a 3-point attempt from text."""
    text = ""
    if isinstance(home_desc, str):
        text += home_desc.upper() + " "
    if isinstance(visitor_desc, str):
        text += visitor_desc.upper()
    if "3PT" in text or "3-PT" in text or "3-POINT" in text:
        return 1
    return 0


def main():
    if not RAW_DB.exists():
        raise FileNotFoundError(f"Database not found at {RAW_DB}. "
                                f"Please put nba.sqlite into the data/ folder.")

    # Connect to SQLite and read play_by_play table
    conn = sqlite3.connect(str(RAW_DB))
    query = "SELECT * FROM play_by_play"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Basic safety: ensure required columns exist
    required_cols = [
        "eventmsgtype", "period", "pctimestring",
        "homedescription", "visitordescription",
        "scoremargin", "game_id", "eventnum",
        "player1_id", "player1_name"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in play_by_play: {missing}")

    # Keep only shot events: 1 = made shot, 2 = missed shot
    shots = df[df["eventmsgtype"].isin([1, 2])].copy()
    if shots.empty:
        raise ValueError("No shot events found with eventmsgtype in [1, 2].")

    # Target label: made shot = 1, missed = 0
    shots["label"] = (shots["eventmsgtype"] == 1).astype(int)

    # Period as integer
    shots["period"] = pd.to_numeric(shots["period"], errors="coerce").fillna(0).astype(int)

    # Game clock in seconds remaining in the period
    shots["seconds_remaining"] = shots["pctimestring"].apply(time_to_seconds)
    # Fill NaN with median to avoid errors
    if shots["seconds_remaining"].isna().all():
        # If everything is NaN, just set a constant
        shots["seconds_remaining"] = 600.0
    else:
        shots["seconds_remaining"].fillna(shots["seconds_remaining"].median(), inplace=True)

    # Score margin numeric
    shots["score_margin_num"] = shots["scoremargin"].apply(parse_score_margin)
    if shots["score_margin_num"].isna().all():
        shots["score_margin_num"] = 0
    else:
        shots["score_margin_num"].fillna(0, inplace=True)

    # Home or visitor offense
    shots["is_home_offense"] = shots.apply(
        lambda r: infer_home_offense(r["homedescription"], r["visitordescription"]),
        axis=1
    )
    # Replace NaN with 0.5 (unknown but numeric)
    shots["is_home_offense"] = shots["is_home_offense"].fillna(0.5)

    # Three-point attempt flag
    shots["is_three"] = shots.apply(
        lambda r: detect_three_point(r["homedescription"], r["visitordescription"]),
        axis=1
    ).astype(int)

    # "Clutch" flag: last 2 minutes of period and close score
    shots["is_clutch"] = (
        (shots["seconds_remaining"] <= 120)
        & (shots["score_margin_num"].abs() <= 5)
    ).astype(int)

    # Clean player name
    shots["player_name"] = shots["player1_name"].fillna("Unknown")

    # Feature columns for the model
    feature_cols = [
        "period",
        "seconds_remaining",
        "score_margin_num",
        "is_home_offense",
        "is_three",
        "is_clutch",
    ]

    # Ensure numeric types
    for c in feature_cols:
        shots[c] = pd.to_numeric(shots[c], errors="coerce")

    # Drop rows where any feature or label is missing
    shots = shots.dropna(subset=feature_cols + ["label"])

    # Final dataset for model
    model_df = shots[
        ["game_id", "eventnum", "player1_id", "player_name",
         "pctimestring", "score", "scoremargin"]  # meta info
        + feature_cols
        + ["label"]
    ].reset_index(drop=True)

    out_path = PROCESSED_DIR / "shots_for_model.parquet"
    model_df.to_parquet(out_path, index=False)
    print(f"Saved processed shots to {out_path}")
    print(f"Total shots: {len(model_df)}")


if __name__ == "__main__":
    main()