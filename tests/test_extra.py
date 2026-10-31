import numpy as np
import pandas as pd
import pytest

from tawseel.model import add_history_features, time_split


def test_history_excludes_same_timestamp_and_preserves_order():
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "A"],
            "created_at": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "failed": [0, 1, 1, 0],
        },
        index=[10, 20, 30, 40],
    )

    result = add_history_features(df)

    # Original row order and index are preserved.
    assert result.index.tolist() == [10, 20, 30, 40]

    # Jan 1 has no previous parcels.
    assert result.loc[20, "customer_prior_parcels"] == 0

    # Both Jan 2 parcels see only Jan 1, not each other.
    assert result.loc[10, "customer_prior_parcels"] == 1
    assert result.loc[30, "customer_prior_parcels"] == 1

    # Jan 3 sees all three earlier parcels.
    assert result.loc[40, "customer_prior_parcels"] == 3
    assert result.loc[40, "customer_prior_fail_rate"] == pytest.approx(2 / 3)


def test_history_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A"],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "failed": [1, 0],
        }
    )

    original = df.copy(deep=True)
    add_history_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_history_is_independent_of_input_order():
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B"],
            "created_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01"]
            ),
            "failed": [1, 0, 1, 0],
        }
    )

    original_result = add_history_features(df).sort_index()
    shuffled_result = add_history_features(
        df.sample(frac=1, random_state=42)
    ).sort_index()

    pd.testing.assert_frame_equal(original_result, shuffled_result)


def test_time_split_keeps_boundary_ties_in_test():
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    train, test = time_split(df, test_fraction=0.2)

    # ceil(5 * 0.2) = 1, but both rows tied at the boundary go to test.
    assert len(train) == 3
    assert len(test) == 2
    assert train["created_at"].max() < test["created_at"].min()

    

@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_time_split_rejects_invalid_fraction(fraction):
    df = pd.DataFrame(
        {"created_at": pd.date_range("2026-01-01", periods=5)}
    )

    with pytest.raises(ValueError):
        time_split(df, test_fraction=fraction)