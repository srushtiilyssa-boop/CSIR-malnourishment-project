"""
data_prep.py
------------
Prepares the child malnutrition dataset for the centralized federated
learning demo.

What it does:
1. Loads the raw dataset (malnutrition_child_dataset_cleaned_v3.csv).
2. Engineers a binary target column `malnourished` using the WHO
   standard cut-off (z-score < -2 on ANY of stunting / wasting /
   underweight z-scores = malnourished). This mirrors how child growth
   is actually classified in nutrition surveillance (DHS/NFHS-style).
3. Drops the three z-score columns + child_bmi_raw from the feature
   set, since they are used to build the label (keeping them in would
   leak the answer straight into the model).
4. Splits the data into N client shards (simulating N health-worker /
   clinic sites each holding their own local data) + one held-out
   test set that nobody trains on, used later by test/compare.py to
   score the federated model.

Run once before starting the server/clients:
    python data_prep.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/malnutrition_child_dataset_cleaned_v3.csv"
N_CLIENTS = 4
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Columns used only to build the label -> must not leak into features
LEAKAGE_COLS = ["stunting_zscore", "wasting_zscore", "underweight_zscore", "child_bmi_raw"]

TARGET_COL = "malnourished"


def build_target(df: pd.DataFrame) -> pd.Series:
    """WHO cut-off: moderate-to-severe malnutrition if ANY of the three
    z-scores falls below -2 standard deviations from the reference median."""
    return (
        (df["stunting_zscore"] < -2)
        | (df["wasting_zscore"] < -2)
        | (df["underweight_zscore"] < -2)
    ).astype(int)


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    df[TARGET_COL] = build_target(df)
    print("Class balance:\n", df[TARGET_COL].value_counts(normalize=True))

    feature_df = df.drop(columns=LEAKAGE_COLS)

    # Held-out test set: never seen by any client during training
    train_df, test_df = train_test_split(
        feature_df, test_size=TEST_SIZE, stratify=feature_df[TARGET_COL], random_state=RANDOM_STATE
    )

    test_df.to_csv("data/test_holdout.csv", index=False)
    print(f"Wrote data/test_holdout.csv ({len(test_df)} rows)")

    # Split remaining training data across N clients (simulating N sites)
    shard_size = len(train_df) // N_CLIENTS
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    for i in range(N_CLIENTS):
        start = i * shard_size
        end = None if i == N_CLIENTS - 1 else (i + 1) * shard_size
        shard = train_df.iloc[start:end]
        path = f"data/client_{i+1}.csv"
        shard.to_csv(path, index=False)
        print(f"Wrote {path} ({len(shard)} rows, "
              f"{shard[TARGET_COL].mean():.3f} malnourished rate)")

    print("\nFeature columns used by clients/model:")
    print([c for c in feature_df.columns if c != TARGET_COL])


if __name__ == "__main__":
    main()
