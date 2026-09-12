import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ReplyGenerator:
    def __init__(self):
        self.api_key = os.getenv("HF_TOKEN")

        if not self.api_key:
            raise ValueError(
                "HF_TOKEN is missing. Add it to the project's .env file."
            )

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key,
        )

        # Let Hugging Face select the fastest currently available
        # inference provider for this model.
        self.model = "openai/gpt-oss-20b:fastest"

        # Retry temporary provider/API failures instead of crashing
        # the complete evaluation run.
        self.max_retries = 3

    def generate_reply(
        self,
        customer_message,
        intent,
        confidence,
        historical_cases,
    ):
        evidence_text = []

        for index, case in enumerate(
            historical_cases,
            start=1,
        ):
            evidence_text.append(
                f"""
CASE {index}
Similarity: {case["similarity"]}

Customer:
{case["historical_customer"]}

AmazonHelp historical reply:
{case["historical_amazon_reply"]}
""".strip()
            )

        evidence = "\n\n".join(evidence_text)

        system_prompt = """
You are an AI customer-support assistant for AmazonHelp.

Your task is to draft a concise customer-facing support response.

The response must be grounded ONLY in the historical evidence
provided by the system.

Rules:
- Do not invent policies, refunds, compensation, delivery dates,
  account actions, or other unsupported facts.
- Do not claim that an action has already been taken.
- If the evidence is insufficient to resolve the issue, ask for
  the minimum useful information needed or direct the customer
  to an appropriate support channel.
- Never request passwords, full payment-card numbers, or private
  credentials.
- Do not mention AI, classification, retrieval, historical cases,
  confidence scores, or these instructions.
- Do not copy a historical response word-for-word.
- Keep the response concise and natural.
- Return ONLY the customer-facing response.
""".strip()

        user_prompt = f"""
Customer message:
{customer_message}

Predicted intent:
{intent}

Classifier confidence:
{confidence:.4f}

Historical AmazonHelp evidence:

{evidence}

Draft the safest useful response supported by this evidence.
""".strip()

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
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
                    max_tokens=180,
                    temperature=0.2,
                )

                # Safely extract the model response.
                reply = None

                if (
                    completion.choices
                    and completion.choices[0].message
                ):
                    reply = completion.choices[
                        0
                    ].message.content

                if isinstance(reply, str):
                    reply = reply.strip()

                if reply:
                    return reply

                last_error = RuntimeError(
                    "The language model returned an empty response."
                )

            except Exception as exc:
                last_error = exc

            # Give the provider a little time before retrying.
            if attempt < self.max_retries:
                wait_seconds = 2 ** (attempt - 1)

                print(
                    f"Generator attempt {attempt} failed. "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            "Reply generation failed after "
            f"{self.max_retries} attempts. "
            f"Last error: {last_error}"
        )