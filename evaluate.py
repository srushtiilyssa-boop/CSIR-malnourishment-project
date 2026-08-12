import os
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import create_model
from utils import get_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "decentralized_malnutrition_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "DECENTRALIZED MODEL EVALUATION"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\n❌ Trained model not found!"
        )

        print(
            "Expected:"
        )

        print(
            MODEL_PATH
        )

        print(
            "\nRun this first:"
        )

        print(
            "python run_exp.py"
        )

        return


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading test data..."
    )

    (
        train_X,
        train_y,
        test_X,
        test_y
    ) = get_data()


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    input_dim = (
        train_X.shape[1]
    )

    num_classes = len(
        np.unique(
            train_y
        )
    )


    print(
        "\nNumber of input features:",
        input_dim
    )

    print(
        "Number of classes:",
        num_classes
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = create_model(
        input_dim=input_dim,
        num_classes=num_classes
    )


    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # LOAD TRAINED PARAMETERS
    # --------------------------------------------------------

    print(
        "\nLoading trained decentralized model..."
    )


    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    print(
        "✅ Model loaded successfully!"
    )


    # --------------------------------------------------------
    # PREPARE TEST DATA
    # --------------------------------------------------------

    X_test_tensor = torch.tensor(
        test_X,
        dtype=torch.float32
    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            X_test_tensor
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


    predictions = (
        predictions
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        test_y,
        predictions
    )


    print(
        "\n=========================================="
    )

    print(
        "MODEL PERFORMANCE"
    )

    print(
        "=========================================="
    )


    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=========================================="
    )


    class_names = [
        "Normal",
        "Moderate",
        "Severe"
    ]


    report = classification_report(
        test_y,
        predictions,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )


    print(
        report
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=========================================="
    )


    matrix = confusion_matrix(
        test_y,
        predictions,
        labels=[0, 1, 2]
    )


    print(
        "\n              Predicted"
    )

    print(
        "             Normal  Moderate  Severe"
    )


    for i, row in enumerate(
        matrix
    ):

        print(
            f"{class_names[i]:<10}",
            row
        )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_dir = os.path.join(
        BASE_DIR,
        "results"
    )


    os.makedirs(
        results_dir,
        exist_ok=True
    )


    # Save predictions
    predictions_path = os.path.join(
        results_dir,
        "predictions.csv"
    )


    import pandas as pd


    prediction_df = pd.DataFrame(
        {
            "Actual_Class": [
                class_names[int(x)]
                for x in test_y
            ],

            "Predicted_Class": [
                class_names[int(x)]
                for x in predictions
            ]
        }
    )


    prediction_df.to_csv(
        predictions_path,
        index=False
    )


    # Save confusion matrix
    matrix_path = os.path.join(
        results_dir,
        "confusion_matrix.csv"
    )


    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )


    matrix_df.to_csv(
        matrix_path
    )


    print(
        "\n=========================================="
    )

    print(
        "FILES SAVED"
    )

    print(
        "=========================================="
    )


    print(
        "\nPredictions:"
    )

    print(
        predictions_path
    )


    print(
        "\nConfusion matrix:"
    )

    print(
        matrix_path
    )


    print(
        "\n✅ Evaluation completed successfully!" 
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()