from classifier import AmazonHelpClassifier
from retrieval import AmazonHelpRetriever
from generator import ReplyGenerator
from escalation import EscalationPolicy


class AmazonHelpPipeline:
    def __init__(self):
        print("Initializing AmazonHelp agent...")

        self.classifier = AmazonHelpClassifier()
        self.retriever = AmazonHelpRetriever()
        self.generator = ReplyGenerator()
        self.escalation = EscalationPolicy()

        print("Building historical retrieval index...")
        self.retriever.build_index()

        print("Agent ready.")

    def analyze(self, customer_message, top_k=5):
        if not customer_message or not customer_message.strip():
            raise ValueError("Customer message cannot be empty.")

        # 1. Classify the customer message
        classification = self.classifier.classify(customer_message)

        # 2. Retrieve historically similar AmazonHelp conversations
        historical_cases = self.retriever.retrieve(
            customer_message,
            top_k=top_k,
        )

        # 3. Generate a grounded reply
        reply = self.generator.generate_reply(
            customer_message=customer_message,
            intent=classification["intent"],
            confidence=classification["confidence"],
            historical_cases=historical_cases,
        )

        # 4. Decide whether the request can be safely auto-handled
        escalation = self.escalation.decide(
            customer_message=customer_message,
            intent=classification["intent"],
            confidence=classification["confidence"],
            historical_cases=historical_cases,
        )

        return {
            "customer_message": customer_message.strip(),
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "intent_scores": classification["scores"],
            "historical_cases": historical_cases,
            "reply": reply,
            "decision": escalation["decision"],
            "escalation_reason": escalation["reason"],
        }


def print_analysis(result):
    print("\n" + "=" * 70)
    print("AMAZONHELP SUPPORT AGENT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE")
    print("-" * 70)
    print(result["customer_message"])

    print("\nPREDICTED INTENT")
    print("-" * 70)
    print(result["intent"])
    print(f"Confidence: {result['confidence']:.4f}")

    print("\nTOP HISTORICAL EVIDENCE")
    print("-" * 70)

    for number, case in enumerate(
        result["historical_cases"],
        start=1,
    ):
        print(f"\n[{number}] Similarity: {case['similarity']:.4f}")

        print("\nHistorical customer:")
        print(case["historical_customer"])

        print("\nHistorical AmazonHelp reply:")
        print(case["historical_amazon_reply"])

        print("-" * 70)

    print("\nGENERATED REPLY")
    print("=" * 70)
    print(result["reply"])

    print("\nESCALATION DECISION")
    print("=" * 70)
    print(result["decision"])

    print("\nREASON")
    print("-" * 70)
    print(result["escalation_reason"])

    print("=" * 70)


def main():
    pipeline = AmazonHelpPipeline()

    message = input(
        "\nEnter a customer message:\n> "
    ).strip()

    result = pipeline.analyze(
        message,
        top_k=5,
    )

    print_analysis(result)


if __name__ == "__main__":
    main()