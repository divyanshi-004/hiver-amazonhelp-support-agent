from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_200_cleaned.csv"
)


# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------

INTENTS = [
    "order_status_tracking",
    "delivery_issue",
    "returns_refunds_replacements",
    "order_changes_cancellations",
    "payments_billing_promotions",
    "account_prime_membership",
    "product_device_support",
    "digital_content",
    "product_seller_availability",
    "general_feedback_non_actionable",
]


class AmazonHelpClassifier:
    """
    Supervised AmazonHelp intent classifier.

    The model is trained on the curated AmazonHelp golden dataset and uses:
        TF-IDF features + Logistic Regression

    This classifier replaces the previous Hugging Face zero-shot model
    for the live pipeline.
    """

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else GOLDEN_DATA_PATH

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found:\n{self.data_path}"
            )

        self.model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        max_features=50000,
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        self._train()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def _train(self):
        df = pd.read_csv(self.data_path)

        required_columns = {
            "customer_text",
            "verified_intent",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                "Golden dataset is missing required columns: "
                + ", ".join(sorted(missing))
            )

        df = df[
            df["customer_text"].notna()
            & df["verified_intent"].notna()
        ].copy()

        df["customer_text"] = df["customer_text"].astype(str).str.strip()
        df["verified_intent"] = (
            df["verified_intent"].astype(str).str.strip()
        )

        df = df[df["customer_text"] != ""]

        # Keep only the intents used by the project.
        df = df[df["verified_intent"].isin(INTENTS)]

        if df.empty:
            raise ValueError(
                "No usable training examples were found in the golden dataset."
            )

        unique_intents = sorted(df["verified_intent"].unique())

        if len(unique_intents) < 2:
            raise ValueError(
                "The golden dataset must contain at least two different intents."
            )

        self.model.fit(
            df["customer_text"],
            df["verified_intent"],
        )

        self.training_examples = len(df)
        self.training_intents = unique_intents

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------

    def classify(self, text):
        if not text or not text.strip():
            raise ValueError("Customer message cannot be empty.")

        text = text.strip()

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.named_steps["classifier"].classes_

        ranked = sorted(
            [
                {
                    "intent": intent,
                    "confidence": round(float(score), 4),
                }
                for intent, score in zip(classes, probabilities)
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "intent": ranked[0]["intent"],
            "confidence": ranked[0]["confidence"],
            "scores": ranked,
        }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def main():
    classifier = AmazonHelpClassifier()

    print("=" * 70)
    print("AMAZONHELP INTENT CLASSIFIER")
    print("=" * 70)

    print(f"\nTraining examples: {classifier.training_examples}")
    print(
        "Intents:",
        ", ".join(classifier.training_intents),
    )

    message = input("\nEnter a customer message:\n> ").strip()

    result = classifier.classify(message)

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULT")
    print("=" * 70)

    print(f"\nPredicted intent: {result['intent']}")
    print(f"Confidence:       {result['confidence']:.4f}")

    print("\nAll intent scores:")

    for item in result["scores"]:
        print(
            f"  {item['intent']:<35} "
            f"{item['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()