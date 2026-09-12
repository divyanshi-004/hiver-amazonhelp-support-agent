from pathlib import Path
import sys
import re
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from escalation import EscalationPolicy


GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_200_cleaned.csv"
)


# -------------------------------------------------------------------
# Independent safety-routing reference rules
# -------------------------------------------------------------------

HIGH_RISK_PATTERNS = [
    r"\bfraud\b",
    r"\bscam\b",
    r"\bstolen\b",
    r"\bidentity theft\b",
    r"\bhacked\b",
    r"\bunauthorized\b",
    r"\bchargeback\b",
    r"\blawsuit\b",
    r"\blegal action\b",
    r"\bpolice\b",
    r"\bthreat\b",
    r"\bdanger\b",
    r"\binjured\b",
    r"\binjury\b",
    r"\bfire\b",
    r"\bexplosion\b",
]

SECURITY_PATTERNS = [
    r"\bpassword\b",
    r"\botp\b",
    r"\bverification code\b",
    r"\bsecurity code\b",
    r"\baccount hacked\b",
    r"\bcan't access my account\b",
    r"\bcannot access my account\b",
    r"\bchanged my email\b",
    r"\bchanged my password\b",
]

EXPLICIT_HUMAN_PATTERNS = [
    r"\bspeak to a manager\b",
    r"\bspeak to a supervisor\b",
    r"\bmanager\b",
    r"\bsupervisor\b",
    r"\bhuman agent\b",
    r"\breal person\b",
    r"\brepresentative\b",
    r"\bperson\b.*\bspeak\b",
]


def matches_any(text, patterns):
    text = text.lower()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def independent_expected_decision(message):
    """
    Independent reference label.

    HUMAN is required for clearly safety-sensitive or
    human-request cases.

    Everything else is AUTO for this safety-routing test.

    This does NOT claim that every ordinary support request
    should truly be auto-handled in production; it is a
    conservative, independently defined benchmark for
    the explicit escalation triggers.
    """

    text = message.lower()

    if matches_any(text, HIGH_RISK_PATTERNS):
        return "HUMAN"

    if matches_any(text, SECURITY_PATTERNS):
        return "HUMAN"

    if matches_any(text, EXPLICIT_HUMAN_PATTERNS):
        return "HUMAN"

    return "AUTO"


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

    print("=" * 70)
    print("ESCALATION SAFETY EVALUATION")
    print("=" * 70)

    print(
        f"Golden examples: {len(df)}"
    )

    print()

    policy = EscalationPolicy()

    expected = []
    predicted = []

    disagreements = []

    for number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        customer_message = str(
            row["customer_text"]
        )

        intent = str(
            row["verified_intent"]
        )

        # We deliberately use a high confidence value here so that
        # this benchmark isolates explicit safety/human-request
        # escalation triggers rather than reproducing the classifier.
        confidence = 1.0

        # No retrieval is required for the independent safety test.
        historical_cases = [
            {
                "similarity": 1.0,
                "historical_customer": "",
                "historical_amazon_reply": "",
            }
        ]

        expected_decision = (
            independent_expected_decision(
                customer_message
            )
        )

        result = policy.decide(
            customer_message=customer_message,
            intent=intent,
            confidence=confidence,
            historical_cases=historical_cases,
        )

        predicted_decision = result["decision"]

        expected.append(expected_decision)
        predicted.append(predicted_decision)

        if expected_decision != predicted_decision:

            disagreements.append(
                {
                    "customer_text": customer_message,
                    "expected": expected_decision,
                    "predicted": predicted_decision,
                    "reason": result["reason"],
                }
            )

        if number % 25 == 0:
            print(
                f"Evaluated: {number}/{len(df)}"
            )

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------

    expected_binary = [
        1 if value == "HUMAN" else 0
        for value in expected
    ]

    predicted_binary = [
        1 if value == "HUMAN" else 0
        for value in predicted
    ]

    accuracy = accuracy_score(
        expected_binary,
        predicted_binary,
    )

    precision = precision_score(
        expected_binary,
        predicted_binary,
        zero_division=0,
    )

    recall = recall_score(
        expected_binary,
        predicted_binary,
        zero_division=0,
    )

    f1 = f1_score(
        expected_binary,
        predicted_binary,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        expected_binary,
        predicted_binary,
        labels=[0, 1],
    ).ravel()

    # False auto-handle:
    # expected HUMAN but system predicted AUTO.
    false_auto_handle_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Human-escalation precision: "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Human-escalation recall: "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Human-escalation F1: "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print()

    print(
        f"True AUTO / Predicted AUTO: {tn}"
    )

    print(
        f"True AUTO / Predicted HUMAN: {fp}"
    )

    print(
        f"True HUMAN / Predicted AUTO: {fn}"
    )

    print(
        f"True HUMAN / Predicted HUMAN: {tp}"
    )

    print()

    print(
        f"False auto-handle rate: "
        f"{false_auto_handle_rate:.4f} "
        f"({false_auto_handle_rate * 100:.2f}%)"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is an independent safety-trigger benchmark. "
        "It evaluates explicit high-risk, account-security, "
        "and human-request escalation triggers."
    )

    print(
        "It is not a fully human-labelled production escalation "
        "benchmark and should be described that way in the report."
    )

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------

    output_dir = PROJECT_ROOT / "evaluation"
    output_dir.mkdir(exist_ok=True)

    results_file = (
        output_dir
        / "escalation_results.txt"
    )

    disagreements_file = (
        output_dir
        / "escalation_disagreements.csv"
    )

    disagreement_df = pd.DataFrame(
        disagreements
    )

    disagreement_df.to_csv(
        disagreements_file,
        index=False,
    )

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ESCALATION SAFETY EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Golden examples: {len(df)}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        f.write(
            f"Human-escalation precision: {precision:.4f}\n"
        )

        f.write(
            f"Human-escalation recall: {recall:.4f}\n"
        )

        f.write(
            f"Human-escalation F1: {f1:.4f}\n"
        )

        f.write(
            f"False auto-handle rate: "
            f"{false_auto_handle_rate:.4f}\n"
        )

        f.write("\n")

        f.write(
            "Reference-label methodology:\n"
            "HUMAN was independently assigned to messages "
            "containing explicit high-risk, account-security, "
            "or explicit human-request triggers. Other examples "
            "were assigned AUTO for this safety-trigger benchmark. "
            "This is a policy benchmark, not a human-labelled "
            "production escalation gold set.\n"
        )

    print()
    print(
        f"Saved summary to:\n"
        f"{results_file}"
    )

    print(
        f"Saved disagreements to:\n"
        f"{disagreements_file}"
    )


if __name__ == "__main__":
    main()