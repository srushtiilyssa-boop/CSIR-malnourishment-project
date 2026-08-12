import os
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "malnutrition_child_dataset_cleaned_v3.csv"
)


# ============================================================
# CREATE NUTRITION CLASS
# ============================================================

def create_nutrition_class(df):
    """
    Create a target class from child growth z-scores.

    Classes:
        0 = Normal
        1 = Moderate
        2 = Severe

    WHO-style z-score thresholds:
        >= -2          -> Normal
        -3 to < -2     -> Moderate
        < -3           -> Severe
    """

    zscore_columns = [
        "stunting_zscore",
        "wasting_zscore",
        "underweight_zscore"
    ]

    missing_columns = [
        col
        for col in zscore_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required z-score columns are missing: "
            + str(missing_columns)
        )

    df = df.copy()

    # Convert z-score columns to numbers
    for col in zscore_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove rows where all three z-scores are missing
    df = df.dropna(
        subset=zscore_columns,
        how="all"
    )

    # Fill missing z-scores with 0
    # 0 represents the normal/reference point.
    df[zscore_columns] = (
        df[zscore_columns]
        .fillna(0)
    )

    # Find the lowest z-score for each child.
    # If any indicator is severely low, the child is Severe.
    minimum_zscore = df[
        zscore_columns
    ].min(axis=1)

    # Default = Normal
    df["Nutrition_Class"] = 0

    # Moderate
    df.loc[
        minimum_zscore < -2,
        "Nutrition_Class"
    ] = 1

    # Severe
    df.loc[
        minimum_zscore < -3,
        "Nutrition_Class"
    ] = 2

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\n==========================================")
    print("LOADING DATASET")
    print("==========================================")

    print(
        "Dataset:",
        DATASET_PATH
    )

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "\nDataset not found!\n"
            "Expected location:\n"
            + DATASET_PATH
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        "\n✅ Dataset loaded successfully!"
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "\nColumn names:"
    )

    for column in df.columns:

        print(
            " -",
            column
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # CREATE TARGET
    # --------------------------------------------------------

    df = create_nutrition_class(
        df
    )

    print(
        "\n=========================================="
    )

    print(
        "NUTRITION CLASS DISTRIBUTION"
    )

    print(
        "=========================================="
    )

    class_counts = (
        df["Nutrition_Class"]
        .value_counts()
        .sort_index()
    )

    class_names = {
        0: "Normal",
        1: "Moderate",
        2: "Severe"
    }

    for class_id, count in (
        class_counts.items()
    ):

        print(
            f"{class_names.get(class_id, 'Unknown')}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # REMOVE TARGET AND Z-SCORES FROM FEATURES
    # --------------------------------------------------------

    target_column = "Nutrition_Class"

    zscore_columns = [
        "stunting_zscore",
        "wasting_zscore",
        "underweight_zscore"
    ]

    columns_to_remove = (
        zscore_columns
        + [target_column]
    )

    feature_df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    # --------------------------------------------------------
    # HANDLE CATEGORICAL COLUMNS
    # --------------------------------------------------------

    categorical_columns = (
        feature_df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    print(
        "\nCategorical columns:"
    )

    print(
        categorical_columns
    )

    if categorical_columns:

        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_columns,
            dummy_na=True
        )

    # --------------------------------------------------------
    # CONVERT EVERYTHING TO NUMERIC
    # --------------------------------------------------------

    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # HANDLE INFINITE VALUES
    # --------------------------------------------------------

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # FILL MISSING VALUES
    # --------------------------------------------------------

    feature_df = feature_df.fillna(
        feature_df.median(
            numeric_only=True
        )
    )

    # Any columns still containing NaN
    # are filled with zero.

    feature_df = feature_df.fillna(
        0
    )

    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(
        feature_df
    )

    X = X.astype(
        np.float32
    )

    y = df[
        target_column
    ].values.astype(
        np.int64
    )

    print(
        "\n=========================================="
    )

    print(
        "DATA PREPARATION COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        "Number of samples:",
        X.shape[0]
    )

    print(
        "Number of features:",
        X.shape[1]
    )

    print(
        "Number of classes:",
        len(
            np.unique(y)
        )
    )

    return X, y


# ============================================================
# GET DATA
# ============================================================

def get_data(args=None):

    df = load_dataset()

    X, y = prepare_data(
        df
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_X, test_X, train_y, test_y = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        "\n=========================================="
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "=========================================="
    )

    print(
        "Training samples:",
        len(train_X)
    )

    print(
        "Testing samples:",
        len(test_X)
    )

    return (
        train_X,
        train_y,
        test_X,
        test_y
    )


# ============================================================
# PARTITION DATA BETWEEN CLIENTS
# ============================================================

def partition_data(
    train_X,
    train_y,
    args
):

    n_clients = args.n_clients

    if n_clients <= 0:

        raise ValueError(
            "Number of clients must be greater than 0."
        )

    if n_clients > len(train_X):

        raise ValueError(
            "Number of clients cannot be greater "
            "than the number of training samples."
        )

    rng = np.random.default_rng(
        getattr(args, "seed", 42)
    )

    # Shuffle indexes
    indexes = np.arange(
        len(train_X)
    )

    rng.shuffle(
        indexes
    )

    # Split indexes between clients
    client_indexes = np.array_split(
        indexes,
        n_clients
    )

    client_data = []

    print(
        "\n=========================================="
    )

    print(
        "DECENTRALIZED CLIENT PARTITION"
    )

    print(
        "=========================================="
    )

    for client_id, indexes in enumerate(
        client_indexes
    ):

        client_X = train_X[
            indexes
        ]

        client_y = train_y[
            indexes
        ]

        client_data.append(
            (
                client_X,
                client_y
            )
        )

        print(
            f"Client {client_id + 1}: "
            f"{len(client_X)} samples"
        )

        unique, counts = np.unique(
            client_y,
            return_counts=True
        )

        distribution = {}

        for class_id, count in zip(
            unique,
            counts
        ):

            distribution[
                int(class_id)
            ] = int(count)

        print(
            "  Class distribution:",
            distribution
        )

    return client_data


# ============================================================
# LOGGER
# ============================================================

def get_logger(
    log_file=None
):

    logger = logging.getLogger(
        "ProxyFL"
    )

    logger.setLevel(
        logging.INFO
    )

    # Avoid duplicate handlers
    if logger.handlers:

        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    if log_file is not None:

        file_handler = (
            logging.FileHandler(
                log_file
            )
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    return logger