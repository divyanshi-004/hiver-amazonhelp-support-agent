from typing import Dict, List


class EscalationPolicy:
    """
    Conservative escalation policy for the AmazonHelp support agent.

    The agent should only auto-handle a request when:
    1. Intent confidence is sufficient.
    2. Historical evidence is sufficiently relevant.
    3. The request does not contain a high-risk or unsupported scenario.

    Otherwise, route to a human with an explicit reason.
    """

    def __init__(
        self,
        min_intent_confidence: float = 0.45,
        min_retrieval_similarity: float = 0.35,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

        self.high_risk_terms = [
            "fraud",
            "scam",
            "stolen",
            "identity theft",
            "hacked",
            "unauthorized",
            "someone accessed my account",
            "credit card fraud",
            "chargeback",
            "lawsuit",
            "legal action",
            "police",
            "threat",
            "danger",
            "injured",
            "injury",
            "fire",
            "explosion",
        ]

        self.account_sensitive_terms = [
            "password",
            "one-time password",
            "otp",
            "verification code",
            "security code",
            "account hacked",
            "login",
            "can't access my account",
        ]

        self.unsupported_signals = [
            "speak to a manager",
            "speak to a supervisor",
            "human agent",
            "real person",
            "representative",
        ]

    def _contains_any(self, text: str, terms: List[str]) -> bool:
        text_lower = text.lower()
        return any(term in text_lower for term in terms)

    def decide(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
        historical_cases: List[Dict],
    ) -> Dict:
        """
        Return an AUTO/HUMAN decision with a human-readable reason.
        """

        message = customer_message.strip()

        # ---------------------------------------------------------
        # 1. High-risk scenarios always go to a human.
        # ---------------------------------------------------------
        if self._contains_any(message, self.high_risk_terms):
            return {
                "decision": "HUMAN",
                "reason": (
                    "High-risk financial, legal, safety, security, "
                    "or fraud-related language requires human review."
                ),
            }

        # ---------------------------------------------------------
        # 2. Account/security-sensitive requests require a human.
        # ---------------------------------------------------------
        if self._contains_any(message, self.account_sensitive_terms):
            return {
                "decision": "HUMAN",
                "reason": (
                    "The request involves account or security-sensitive "
                    "information that should not be resolved automatically."
                ),
            }

        # ---------------------------------------------------------
        # 3. Explicit requests for human assistance.
        # ---------------------------------------------------------
        if self._contains_any(message, self.unsupported_signals):
            return {
                "decision": "HUMAN",
                "reason": (
                    "The customer explicitly requested human assistance."
                ),
            }

        # ---------------------------------------------------------
        # 4. Low classifier confidence.
        # ---------------------------------------------------------
        if confidence < self.min_intent_confidence:
            return {
                "decision": "HUMAN",
                "reason": (
                    f"Intent confidence ({confidence:.2f}) is below the "
                    f"safe auto-handling threshold "
                    f"({self.min_intent_confidence:.2f})."
                ),
            }

        # ---------------------------------------------------------
        # 5. No useful historical evidence.
        # ---------------------------------------------------------
        if not historical_cases:
            return {
                "decision": "HUMAN",
                "reason": (
                    "No historical AmazonHelp examples were available "
                    "to ground the response."
                ),
            }

        # ---------------------------------------------------------
        # 6. Weak historical evidence.
        # ---------------------------------------------------------
        best_similarity = max(
            float(case.get("similarity", 0.0))
            for case in historical_cases
        )

        if best_similarity < self.min_retrieval_similarity:
            return {
                "decision": "HUMAN",
                "reason": (
                    f"The strongest historical match has low similarity "
                    f"({best_similarity:.2f}), so there is insufficient "
                    "evidence for safe automatic handling."
                ),
            }

        # ---------------------------------------------------------
        # 7. Otherwise, allow automatic handling.
        # ---------------------------------------------------------
        return {
            "decision": "AUTO",
            "reason": (
                f"Intent confidence ({confidence:.2f}) and historical "
                f"evidence ({best_similarity:.2f}) meet the current "
                "auto-handling thresholds, with no high-risk signals."
            ),
        }