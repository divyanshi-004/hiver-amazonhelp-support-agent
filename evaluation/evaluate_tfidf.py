from pathlib import Path
import sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_200_cleaned.csv"
)


def main():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found:\n{GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required = {
        "customer_text",
        "verified_intent",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            "customer_text",
            "verified_intent",
        ]
    ).copy()

    X = df["customer_text"].astype(str)
    y = df["verified_intent"].astype(str)

    print("=" * 70)
    print("FAIR TF-IDF + LOGISTIC REGRESSION EVALUATION")
    print("=" * 70)

    print(f"Evaluation examples: {len(df)}")
    print()

    # Train on all examples EXCEPT the one being evaluated.
    # This gives us a clean leave-one-out evaluation across
    # the exact same 200-example golden set used by the HF model.
    predictions = []
    truths = []

    for i in range(len(df)):
        X_train = X.drop(index=i)
        y_train = y.drop(index=i)

        X_test = X.iloc[[i]]
        y_test = y.iloc[[i]]

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=50000,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)[0]

        predictions.append(prediction)
        truths.append(y_test.iloc[0])

        if (i + 1) % 10 == 0:
            print(
                f"Evaluated: {i + 1}/{len(df)}"
            )

    accuracy = accuracy_score(
        truths,
        predictions,
    )

    macro_f1 = f1_score(
        truths,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        truths,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        truths,
        predictions,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1: {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print()
    print("PER-INTENT RESULTS")
    print("=" * 70)
    print(report)

    print()
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(matrix)

    output_dir = PROJECT_ROOT / "evaluation"
    output_dir.mkdir(exist_ok=True)

    results_file = (
        output_dir
        / "tfidf_fair_results.txt"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "FAIR TF-IDF + LOGISTIC REGRESSION EVALUATION\n"
        )

        f.write("=" * 70 + "\n")

        f.write(
            f"Evaluation examples: {len(df)}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        f.write(
            f"Macro F1: {macro_f1:.4f}\n\n"
        )

        f.write(
            "PER-INTENT RESULTS\n"
        )

        f.write("=" * 70 + "\n")

        f.write(report)

        f.write(
            "\n\n"
            "Evaluation protocol:\n"
            "Leave-one-out evaluation over the exact same "
            "200-example golden set used for the HF zero-shot "
            "classifier. For each example, the model was trained "
            "on the other 199 examples and evaluated on the "
            "held-out example.\n"
        )

    print()
    print(
        f"Saved results to:\n{results_file}"
    )


if __name__ == "__main__":
    main()