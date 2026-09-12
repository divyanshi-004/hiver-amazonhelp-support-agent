from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import AmazonHelpRetriever
from classifier import AmazonHelpClassifier


GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_200_cleaned.csv"
)


def extract_customer_text(result):
    """
    Extract the historical customer message from whatever key
    AmazonHelpRetriever currently uses.

    This keeps the evaluation script compatible with the existing
    retriever implementation instead of forcing unnecessary changes
    to the retrieval engine.
    """

    possible_keys = [
        "customer_text",
        "customer",
        "text",
        "query",
        "historical_customer",
        "historical_customer_text",
        "document",
    ]

    for key in possible_keys:
        value = result.get(key)

        if value is not None and str(value).strip():
            return str(value)

    return None


def main():

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found:\n{GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required = {
        "customer_text",
        "verified_intent"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in golden dataset: {sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            "customer_text",
            "verified_intent"
        ]
    ).copy()

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Golden examples: {len(df)}"
    )

    print()

    retriever = AmazonHelpRetriever()

    print("Building retrieval index...")
    retriever.build_index()

    print()

    classifier = AmazonHelpClassifier()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    evaluated = 0
    failed_retrievals = 0

    total = len(df)

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        query = str(
            row["customer_text"]
        )

        true_intent = str(
            row["verified_intent"]
        )

        results = retriever.retrieve(
            query,
            top_k=5
        )

        retrieved_intents = []

        for result in results:

            historical_text = extract_customer_text(
                result
            )

            if historical_text is None:

                failed_retrievals += 1

                print(
                    "\nWARNING: Could not identify "
                    "historical customer text."
                )

                print(
                    "Retriever returned:"
                )

                print(
                    result
                )

                continue

            try:

                classification = (
                    classifier.classify(
                        historical_text
                    )
                )

                retrieved_intents.append(
                    classification["intent"]
                )

            except Exception as exc:

                print(
                    f"\nWarning: classifier failed for "
                    f"golden example {row_number}: {exc}"
                )

        if not retrieved_intents:

            continue

        evaluated += 1

        if true_intent in retrieved_intents[:1]:

            hits_at_1 += 1

        if true_intent in retrieved_intents[:3]:

            hits_at_3 += 1

        if true_intent in retrieved_intents[:5]:

            hits_at_5 += 1

        if evaluated % 10 == 0:

            print(
                f"Evaluated: "
                f"{evaluated}/{total}"
            )

    if evaluated == 0:

        raise RuntimeError(
            "No examples could be evaluated."
        )

    recall_at_1 = (
        hits_at_1 / evaluated
    )

    recall_at_3 = (
        hits_at_3 / evaluated
    )

    recall_at_5 = (
        hits_at_5 / evaluated
    )

    print()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Recall@1: "
        f"{recall_at_1:.4f} "
        f"({recall_at_1 * 100:.2f}%)"
    )

    print(
        f"Recall@3: "
        f"{recall_at_3:.4f} "
        f"({recall_at_3 * 100:.2f}%)"
    )

    print(
        f"Recall@5: "
        f"{recall_at_5:.4f} "
        f"({recall_at_5 * 100:.2f}%)"
    )

    print()

    print(
        f"Successfully evaluated: "
        f"{evaluated}/{total}"
    )

    print(
        f"Unrecognized retrieval results: "
        f"{failed_retrievals}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Recall@K here uses agreement between the "
        "golden intent and the intent predicted for "
        "retrieved historical messages as a proxy "
        "for retrieval relevance."
    )

    print(
        "This is a proxy metric, not human-annotated "
        "retrieval relevance."
    )

    output_dir = (
        PROJECT_ROOT / "evaluation"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir
        / "retrieval_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "RETRIEVAL EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Golden examples: "
            f"{len(df)}\n"
        )

        f.write(
            f"Evaluated examples: "
            f"{evaluated}\n"
        )

        f.write(
            f"Recall@1: "
            f"{recall_at_1:.4f}\n"
        )

        f.write(
            f"Recall@3: "
            f"{recall_at_3:.4f}\n"
        )

        f.write(
            f"Recall@5: "
            f"{recall_at_5:.4f}\n"
        )

        f.write(
            f"Unrecognized retrieval results: "
            f"{failed_retrievals}\n\n"
        )

        f.write(
            "Metric definition:\n"
        )

        f.write(
            "Intent agreement between the golden "
            "query and retrieved historical customer "
            "messages is used as a proxy for "
            "retrieval relevance. This is not "
            "human-annotated relevance.\n"
        )

    print(
        f"Saved results to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()