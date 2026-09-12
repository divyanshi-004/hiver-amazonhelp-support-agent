from pathlib import Path
import sys
import json
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Project setup
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from classifier import AmazonHelpClassifier
from retrieval import AmazonHelpRetriever
from generator import ReplyGenerator


GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "amazonhelp_golden_200_cleaned.csv"
)

MODEL = "openai/gpt-oss-20b:fastest"


# -------------------------------------------------------------------
# Judge
# -------------------------------------------------------------------

class ReplyJudge:
    def __init__(self):
        api_key = __import__("os").getenv("HF_TOKEN")

        if not api_key:
            raise ValueError(
                "HF_TOKEN is missing. Add it to the project's .env file."
            )

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )

    def judge(
        self,
        customer_message,
        predicted_intent,
        generated_reply,
        historical_cases,
    ):
        evidence_blocks = []

        for i, case in enumerate(
            historical_cases,
            start=1,
        ):
            evidence_blocks.append(
                f"""
CASE {i}
Similarity: {case["similarity"]}

Historical customer:
{case["historical_customer"]}

Historical AmazonHelp reply:
{case["historical_amazon_reply"]}
""".strip()
            )

        evidence = "\n\n".join(evidence_blocks)

        system_prompt = """
You are a strict evaluator of an AI customer-support reply.

Evaluate ONLY the generated reply using:
1. the customer's message
2. the predicted intent
3. the retrieved historical evidence

Score each dimension from 0 to 2:

groundedness:
0 = introduces unsupported facts/actions/policies
1 = mostly grounded but contains a minor unsupported element
2 = fully grounded in the supplied evidence

correctness:
0 = clearly inappropriate or materially wrong
1 = partly appropriate but incomplete or questionable
2 = appropriate and consistent with the evidence

relevance:
0 = does not address the customer's problem
1 = partially addresses it
2 = directly addresses the customer's problem

completeness:
0 = does not provide a useful next step
1 = somewhat useful but misses an important supported next step
2 = gives the safest useful response supported by the evidence

tone:
0 = inappropriate, rude, robotic, or unsafe
1 = acceptable but awkward
2 = concise, natural, professional support tone

Also determine whether the generated reply contains any
unsupported claim that is important enough to be considered
a grounding failure.

Return ONLY valid JSON with this exact structure:

{
  "groundedness": 0,
  "correctness": 0,
  "relevance": 0,
  "completeness": 0,
  "tone": 0,
  "unsupported_claim": false,
  "reason": "brief explanation"
}
""".strip()

        user_prompt = f"""
Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

Generated reply:
{generated_reply}

Retrieved historical evidence:
{evidence}

Evaluate the generated reply strictly against the supplied evidence.
Do not reward information that is merely plausible if it is not supported
by the evidence.
""".strip()

        last_error = None

        # Retry judge calls because Hugging Face can temporarily time out.
        for attempt in range(1, 4):
            try:
                completion = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    max_tokens=300,
                    temperature=0,
                )

                if not completion.choices:
                    raise RuntimeError(
                        "Judge returned no choices."
                    )

                text = completion.choices[0].message.content

                if not text:
                    raise RuntimeError(
                        "Judge returned an empty response."
                    )

                return self._parse_json(text)

            except Exception as exc:
                last_error = exc

                if attempt < 3:
                    wait_seconds = 2 ** (attempt - 1)

                    print(
                        f"Judge attempt {attempt} failed. "
                        f"Retrying in {wait_seconds}s..."
                    )

                    import time
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"Judge failed after 3 attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_json(text):
        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if match:
            try:
                return json.loads(
                    match.group(0)
                )

            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse judge JSON:\n{text}"
        )


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def main():

    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found:\n{GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required_columns = {
        "customer_text",
        "verified_intent",
    }

    missing = required_columns - set(df.columns)

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

    # ---------------------------------------------------------------
    # Select 5 examples per intent = 50 total.
    # ---------------------------------------------------------------

    evaluation_parts = []

    for intent in sorted(
        df["verified_intent"].astype(str).unique()
    ):
        subset = df[
            df["verified_intent"].astype(str) == intent
        ].head(5)

        evaluation_parts.append(subset)

    eval_df = pd.concat(
        evaluation_parts,
        ignore_index=True,
    )

    print("=" * 70)
    print("REPLY QUALITY EVALUATION")
    print("=" * 70)
    print(
        f"Evaluation examples: {len(eval_df)}"
    )
    print()

    # ---------------------------------------------------------------
    # Initialize components.
    # ---------------------------------------------------------------

    classifier = AmazonHelpClassifier()
    retriever = AmazonHelpRetriever()
    generator = ReplyGenerator()
    judge = ReplyJudge()

    print("Building retrieval index...")
    retriever.build_index()
    print("Ready.")
    print()

    results = []

    generation_failures = 0
    judge_failures = 0

    # ---------------------------------------------------------------
    # Run generation + judging.
    # ---------------------------------------------------------------

    for number, (_, row) in enumerate(
        eval_df.iterrows(),
        start=1,
    ):

        customer_message = str(
            row["customer_text"]
        )

        true_intent = str(
            row["verified_intent"]
        )

        print(
            f"[{number}/{len(eval_df)}] "
            f"{customer_message[:100]}"
        )

        # -----------------------------------------------------------
        # Classification
        # -----------------------------------------------------------

        try:
            classification = classifier.classify(
                customer_message
            )

        except Exception as exc:
            print(
                f"  CLASSIFIER FAILURE: {exc}"
            )

            results.append(
                {
                    "customer_tweet_id": row.get(
                        "customer_tweet_id",
                        "",
                    ),
                    "customer_text": customer_message,
                    "true_intent": true_intent,
                    "predicted_intent": "",
                    "confidence": "",
                    "generated_reply": "",
                    "groundedness": "",
                    "correctness": "",
                    "relevance": "",
                    "completeness": "",
                    "tone": "",
                    "unsupported_claim": "",
                    "judge_reason": "",
                    "generation_failed": False,
                    "judge_failed": False,
                    "classifier_failed": True,
                    "failure_reason": str(exc),
                }
            )

            continue

        predicted_intent = classification["intent"]
        confidence = classification["confidence"]

        # -----------------------------------------------------------
        # Retrieval
        # -----------------------------------------------------------

        try:
            historical_cases = retriever.retrieve(
                customer_message,
                top_k=5,
            )

        except Exception as exc:
            print(
                f"  RETRIEVAL FAILURE: {exc}"
            )

            results.append(
                {
                    "customer_tweet_id": row.get(
                        "customer_tweet_id",
                        "",
                    ),
                    "customer_text": customer_message,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "confidence": confidence,
                    "generated_reply": "",
                    "groundedness": "",
                    "correctness": "",
                    "relevance": "",
                    "completeness": "",
                    "tone": "",
                    "unsupported_claim": "",
                    "judge_reason": "",
                    "generation_failed": False,
                    "judge_failed": False,
                    "classifier_failed": False,
                    "retrieval_failed": True,
                    "failure_reason": str(exc),
                }
            )

            continue

        # -----------------------------------------------------------
        # Generation
        # -----------------------------------------------------------

        try:
            generated_reply = generator.generate_reply(
                customer_message=customer_message,
                intent=predicted_intent,
                confidence=confidence,
                historical_cases=historical_cases,
            )

        except Exception as exc:

            generation_failures += 1

            print(
                f"  GENERATION FAILURE: {exc}"
            )

            results.append(
                {
                    "customer_tweet_id": row.get(
                        "customer_tweet_id",
                        "",
                    ),
                    "customer_text": customer_message,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "confidence": confidence,
                    "generated_reply": "",
                    "groundedness": "",
                    "correctness": "",
                    "relevance": "",
                    "completeness": "",
                    "tone": "",
                    "unsupported_claim": "",
                    "judge_reason": "",
                    "generation_failed": True,
                    "judge_failed": False,
                    "classifier_failed": False,
                    "retrieval_failed": False,
                    "failure_reason": str(exc),
                }
            )

            continue

        # -----------------------------------------------------------
        # Judge
        # -----------------------------------------------------------

        try:
            judge_result = judge.judge(
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                generated_reply=generated_reply,
                historical_cases=historical_cases,
            )

        except Exception as exc:

            judge_failures += 1

            print(
                f"  JUDGE FAILURE: {exc}"
            )

            results.append(
                {
                    "customer_tweet_id": row.get(
                        "customer_tweet_id",
                        "",
                    ),
                    "customer_text": customer_message,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "confidence": confidence,
                    "generated_reply": generated_reply,
                    "groundedness": "",
                    "correctness": "",
                    "relevance": "",
                    "completeness": "",
                    "tone": "",
                    "unsupported_claim": "",
                    "judge_reason": "",
                    "generation_failed": False,
                    "judge_failed": True,
                    "classifier_failed": False,
                    "retrieval_failed": False,
                    "failure_reason": str(exc),
                }
            )

            continue

        # -----------------------------------------------------------
        # Successful example
        # -----------------------------------------------------------

        print(
            f"  True intent: {true_intent}"
        )

        print(
            f"  Predicted intent: "
            f"{predicted_intent}"
        )

        print(
            f"  Generated reply: "
            f"{generated_reply[:160]}"
        )

        print(
            f"  Scores: "
            f"G={judge_result['groundedness']} "
            f"C={judge_result['correctness']} "
            f"R={judge_result['relevance']} "
            f"Co={judge_result['completeness']} "
            f"T={judge_result['tone']}"
        )

        print(
            f"  Unsupported claim: "
            f"{judge_result['unsupported_claim']}"
        )

        print()

        results.append(
            {
                "customer_tweet_id": row.get(
                    "customer_tweet_id",
                    "",
                ),
                "customer_text": customer_message,
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "generated_reply": generated_reply,
                "groundedness": judge_result[
                    "groundedness"
                ],
                "correctness": judge_result[
                    "correctness"
                ],
                "relevance": judge_result[
                    "relevance"
                ],
                "completeness": judge_result[
                    "completeness"
                ],
                "tone": judge_result[
                    "tone"
                ],
                "unsupported_claim": judge_result[
                    "unsupported_claim"
                ],
                "judge_reason": judge_result[
                    "reason"
                ],
                "generation_failed": False,
                "judge_failed": False,
                "classifier_failed": False,
                "retrieval_failed": False,
                "failure_reason": "",
            }
        )

    # ---------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    dimensions = [
        "groundedness",
        "correctness",
        "relevance",
        "completeness",
        "tone",
    ]

    successful_judged = results_df[
        results_df["judge_failed"] == False
    ].copy()

    successful_judged = successful_judged[
        successful_judged["generated_reply"].astype(str).str.strip() != ""
    ]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Total evaluation examples: "
        f"{len(eval_df)}"
    )

    print(
        f"Successfully judged replies: "
        f"{len(successful_judged)}"
    )

    print(
        f"Generation failures: "
        f"{generation_failures}"
    )

    print(
        f"Judge failures: "
        f"{judge_failures}"
    )

    if len(eval_df) > 0:
        generation_failure_rate = (
            generation_failures / len(eval_df)
        )

        print(
            f"Generation failure rate: "
            f"{generation_failure_rate:.3f} "
            f"({generation_failure_rate * 100:.1f}%)"
        )

    if not successful_judged.empty:

        for dimension in dimensions:

            mean_score = successful_judged[
                dimension
            ].astype(float).mean()

            print(
                f"{dimension}: "
                f"{mean_score:.3f}/2.000 "
                f"({mean_score / 2 * 100:.1f}%)"
            )

        unsupported_rate = (
            successful_judged[
                "unsupported_claim"
            ]
            .astype(bool)
            .mean()
        )

        overall_mean = successful_judged[
            dimensions
        ].astype(float).mean().mean()

        print()

        print(
            f"Overall quality score: "
            f"{overall_mean:.3f}/2.000 "
            f"({overall_mean / 2 * 100:.1f}%)"
        )

        print(
            f"Unsupported-claim rate among judged replies: "
            f"{unsupported_rate:.3f} "
            f"({unsupported_rate * 100:.1f}%)"
        )

    else:

        print(
            "No successfully judged replies were available "
            "for quality aggregation."
        )

    # ---------------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "evaluation"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    csv_path = (
        output_dir
        / "reply_quality_results.csv"
    )

    txt_path = (
        output_dir
        / "reply_quality_results.txt"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "REPLY QUALITY EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"Total evaluation examples: "
            f"{len(eval_df)}\n"
        )

        f.write(
            f"Successfully judged replies: "
            f"{len(successful_judged)}\n"
        )

        f.write(
            f"Generation failures: "
            f"{generation_failures}\n"
        )

        f.write(
            f"Judge failures: "
            f"{judge_failures}\n"
        )

        if len(eval_df) > 0:

            generation_failure_rate = (
                generation_failures
                / len(eval_df)
            )

            f.write(
                f"Generation failure rate: "
                f"{generation_failure_rate:.4f}\n"
            )

        f.write("\n")

        if not successful_judged.empty:

            for dimension in dimensions:

                mean_score = successful_judged[
                    dimension
                ].astype(float).mean()

                f.write(
                    f"{dimension}: "
                    f"{mean_score:.3f}/2.000 "
                    f"({mean_score / 2 * 100:.1f}%)\n"
                )

            unsupported_rate = (
                successful_judged[
                    "unsupported_claim"
                ]
                .astype(bool)
                .mean()
            )

            overall_mean = successful_judged[
                dimensions
            ].astype(float).mean().mean()

            f.write(
                f"\nOverall quality score: "
                f"{overall_mean:.3f}/2.000 "
                f"({overall_mean / 2 * 100:.1f}%)\n"
            )

            f.write(
                f"Unsupported-claim rate among judged replies: "
                f"{unsupported_rate:.4f}\n"
            )

        f.write(
            "\nMethodology:\n"
            "Five examples per intent were sampled from the "
            "200-example evaluation set. Each successful generated "
            "reply was evaluated by an LLM judge against the "
            "customer message and top-5 retrieved historical "
            "AmazonHelp cases. Each quality dimension was scored "
            "from 0 to 2. Generation and judge failures were "
            "reported separately rather than silently excluded.\n"
        )

    print()
    print(
        f"Detailed results saved to:\n"
        f"{csv_path}"
    )

    print(
        f"Summary saved to:\n"
        f"{txt_path}"
    )


if __name__ == "__main__":
    main()