from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


TARGET = "failed"

# Only information available when the parcel is created.
# Do NOT add parcel_id, failed, delivered_at, or courier_notes.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


HISTORY_COLUMNS = [
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer history using only parcels with strictly earlier timestamps.

    Parcels belonging to the same customer at the same timestamp do not
    count as prior parcels for one another.
    """
    required = {"customer_id", "created_at", TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    # Aggregate all parcels for each customer at each timestamp.
    grouped = (
        df.groupby(
            ["customer_id", "created_at"],
            dropna=False,
            sort=True,
        )[TARGET]
        .agg(["size", "sum"])
        .rename(columns={"size": "parcel_count", "sum": "failed_count"})
    )

    # Cumulative totals minus the current timestamp group.
    # This excludes all parcels with the same timestamp.
    grouped["prior_parcels"] = (
        grouped.groupby(level=0, sort=False)["parcel_count"].cumsum()
        - grouped["parcel_count"]
    )

    grouped["prior_failures"] = (
        grouped.groupby(level=0, sort=False)["failed_count"].cumsum()
        - grouped["failed_count"]
    )

    # Look up the history for every original row without changing its order.
    row_keys = pd.MultiIndex.from_arrays(
        [df["customer_id"], df["created_at"]],
        names=["customer_id", "created_at"],
    )

    prior_parcels = (
        grouped["prior_parcels"]
        .reindex(row_keys)
        .to_numpy(dtype=float)
    )

    prior_failures = (
        grouped["prior_failures"]
        .reindex(row_keys)
        .to_numpy(dtype=float)
    )

    result["customer_prior_parcels"] = prior_parcels.astype("int64")

    result["customer_prior_fail_rate"] = np.divide(
        prior_failures,
        prior_parcels,
        out=np.full(len(df), np.nan, dtype=float),
        where=prior_parcels > 0,
    )

    return result


def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically.

    The test set starts at the timestamp of the first row among the last
    ceil(len(df) * test_fraction) sorted rows. All rows tied at that
    timestamp are placed in the test set.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist.")

    if df.empty:
        return df.copy(), df.copy()

    n_test = math.ceil(len(df) * test_fraction)

    sorted_df = df.sort_values(time_col, kind="mergesort")
    boundary = sorted_df[time_col].iloc[-n_test]

    train_df = df.loc[df[time_col] < boundary].copy()
    test_df = df.loc[df[time_col] >= boundary].copy()

    return train_df, test_df


def _extract_datetime_features(X: pd.DataFrame) -> np.ndarray:
    """Convert created_at into numeric calendar features."""
    dates = pd.to_datetime(X.iloc[:, 0], errors="coerce")

    features = pd.DataFrame(
        {
            "hour": dates.dt.hour,
            "day_of_week": dates.dt.dayofweek,
            "day_of_year": dates.dt.dayofyear,
            "month": dates.dt.month,
        }
    )

    return features.to_numpy(dtype=float)


def build_pipeline(seed: int = 0) -> Pipeline:
    """
    Build an unfitted preprocessing + Logistic Regression pipeline.

    Extra columns are ignored by the ColumnTransformer.
    """

    numeric_columns = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    categorical_columns = [
        "customer_id",
        "merchant_id",
        "area",
        "promised_slot",
        "address_text",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    datetime_pipeline = Pipeline(
        steps=[
            (
                "extract",
                FunctionTransformer(
                    _extract_datetime_features,
                    validate=False,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("datetime", datetime_pipeline, ["created_at"]),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=seed,
                    max_iter=2000,
                    C=1.0,
                ),
            ),
        ]
    )

    return pipeline


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate a fitted model using its failure probabilities."""
    y_true = test_df[TARGET].to_numpy()
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(test_df)),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """
    Compare Logistic Regression against a prior-probability dummy baseline.

    train_df and test_df should already contain the history features
    returned by add_history_features().
    """
    required_features = FEATURE_COLUMNS + HISTORY_COLUMNS

    missing_train = set(required_features) - set(train_df.columns)
    missing_test = set(required_features) - set(test_df.columns)

    if missing_train or missing_test:
        raise ValueError(
            "Add history features before calling compare_to_baseline(). "
            f"Missing from train: {sorted(missing_train)}; "
            f"missing from test: {sorted(missing_test)}"
        )

    X_train = train_df[required_features]
    y_train = train_df[TARGET]

    model = build_pipeline(seed=seed)
    model.fit(X_train, y_train)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df, y_train)

    model_results = evaluate(model, test_df)
    dummy_results = evaluate(dummy, test_df)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": float(
            model_results["pr_auc"] - dummy_results["pr_auc"]
        ),
    }