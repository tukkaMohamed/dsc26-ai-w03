from tawseel.data import make_parcels
from tawseel.model import (
    add_history_features,
    time_split,
    compare_to_baseline,
)

# Generate the required dataset.
df = make_parcels(n=4000, seed=0)

# Calculate point-in-time history before splitting.
df = add_history_features(df)

# Last 20% by time is the test set.
train_df, test_df = time_split(df, test_fraction=0.2)

# Compare the model with the dummy baseline.
results = compare_to_baseline(
    train_df=train_df,
    test_df=test_df,
    seed=0,
)

print("\nDataset information")
print("-------------------")
print("Total rows:", len(df))
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Failure rate:", round(df["failed"].mean(), 4))

print("\nEvaluation results")
print("------------------")

for model_name in ["dummy", "model"]:
    print(f"\n{model_name.upper()}")
    for metric, value in results[model_name].items():
        print(f"{metric}: {value}")

print("\nPR-AUC lift:", results["lift_pr_auc"])