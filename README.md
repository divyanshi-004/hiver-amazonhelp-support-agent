# Hiver SDE Intern Take-Home — AmazonHelp Support Agent

An AI customer-support agent built using the **Customer Support on Twitter** dataset and historical **AmazonHelp** conversations.

The system is designed around one principle:

> **Optimize for safe, evidence-backed assistance—not maximum automation.**

---

## 1. Problem

Given a new AmazonHelp customer message, the agent performs four steps:

1. **Intent classification** — identify the customer's primary support intent.

2. **Historical retrieval** — retrieve similar AmazonHelp conversations and historical replies.

3. **Escalation decision** — determine whether the case can be safely auto-handled or should be routed to a human.

4. **Response generation** — only for cases approved for AUTO handling, draft a concise reply grounded in retrieved historical evidence.


### Agent Flow

```text
Customer message
       |
       v
+---------------------------+
| Intent Classification     |
| TF-IDF + Logistic         |
| Regression                |
+-------------+-------------+
              |
              v
+---------------------------+
| Historical Retrieval      |
| TF-IDF + Cosine Similarity|
+-------------+-------------+
              |
              v
+---------------------------+
| Escalation Policy         |
| Confidence + Evidence    |
| + Safety Rules            |
+-------------+-------------+
              |
        +-----+-----+
        |           |
      HUMAN        AUTO
        |           |
        |           v
        |    +------------------+
        |    | Grounded LLM     |
        |    | Response         |
        |    +--------+---------+
        |             |
        +-------------+
              |
              v
       Final support action
The escalation decision is made before external LLM generation. High-risk, security-sensitive, low-confidence, or weak-evidence cases can therefore be routed to a human without relying on the external generation provider.

2. What “Good” Means
Dimension	What good means
Intent classification	Correct operational support category
Historical retrieval	Relevant prior AmazonHelp evidence
Response generation	Concise response grounded in available evidence
Escalation	Avoid unsupported or unsafe autonomous handling

The prototype is designed to favor safe handling over maximum automation.

3. Dataset
Source Dataset

The project uses the Customer Support on Twitter dataset.

The prototype is restricted to AmazonHelp to keep the historical support corpus, taxonomy, and response style internally consistent.

The processed AmazonHelp retrieval corpus contains:

100,503 usable historical conversations
historical customer messages
corresponding historical AmazonHelp replies where available

The full processed historical CSV is intentionally not committed to GitHub because of its size.

Expected local path:

data/processed/amazonhelp_direct_interactions.csv

Place the processed historical dataset at that path before running the full support agent.

Golden Evaluation Set

The project includes:

data/golden/amazonhelp_golden_200_cleaned.csv

The evaluation set contains 200 human-verified examples across the project's 10 intents.

Initial candidate labels were reviewed against the customer's primary support need, and the final `verified_intent` labels were confirmed manually.

4. Intent Taxonomy

The agent uses 10 operational intents based on the customer's primary support need.

Intent	Description
order_status_tracking	Order and shipment status or tracking
delivery_issue	Late, missing, failed, or incorrectly recorded delivery
returns_refunds_replacements	Returns, refunds, and replacements
order_changes_cancellations	Order changes and cancellations
payments_billing_promotions	Payments, charges, billing, and promotions
account_prime_membership	Account and Prime membership issues
product_device_support	Physical Amazon device support
digital_content	Digital content and digital services
product_seller_availability	Product, seller, stock, and purchasing availability
general_feedback_non_actionable	Non-actionable feedback and general comments
Taxonomy Boundaries

The taxonomy follows the customer's primary problem or requested outcome, rather than individual keywords.

Examples:

"Where is my package?"
→ order_status_tracking

"My package says delivered but I never received it."
→ delivery_issue

"I want to cancel my order."
→ order_changes_cancellations

"I returned it. Where is my refund?"
→ returns_refunds_replacements

"My payment was declined."
→ payments_billing_promotions

"My Echo won't connect to Wi-Fi."
→ product_device_support

"My Kindle book won't download."
→ digital_content

Escalation is deliberately kept separate from intent classification.

5. Project Structure
hiver-amazonhelp-support-agent/
|
├── data/
│   └── golden/
│       └── amazonhelp_golden_200_cleaned.csv
|
├── evaluation/
│   ├── evaluate_tfidf.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_escalation.py
│   ├── evaluate_replies.py
│   ├── classifier_results.txt
│   ├── escalation_disagreements.csv
│   ├── escalation_results.txt
│   ├── retrieval_results.txt
│   ├── tfidf_fair_results.txt
│   ├── reply_quality_results.csv
│   └── reply_quality_results.txt
|
├── src/
│   ├── classifier.py
│   ├── retrieval.py
│   ├── generator.py
│   ├── escalation.py
│   └── pipeline.py
|
├── check_hf_models.py
├── requirements.txt
├── README.md
└── .gitignore

6. System Components
6.1 Intent Classification

The primary classifier is:

TF-IDF + Logistic Regression

The classifier is trained against the project's human-verified intent set and returns:

predicted intent
confidence score
ranked intent scores

A Hugging Face zero-shot classifier was also evaluated as a baseline.


6.2 Historical Retrieval

Historical AmazonHelp customer messages are indexed using:

TF-IDF
n-grams: 1–2
cosine similarity

The retriever returns the most similar historical customer-support interactions and their associated AmazonHelp replies.

6.3 Response Generation

An external LLM is used to draft a customer-facing response only after the escalation policy allows automatic handling.

The generator is instructed to:

use retrieved historical evidence
avoid unsupported policies or actions
avoid inventing refunds, dates, or guarantees
never claim an action has already occurred when it has not
avoid requesting passwords or private credentials
remain concise and customer-facing
6.4 Escalation

The policy produces either:

AUTO

or:

HUMAN

with an explicit reason.

HUMAN escalation can be triggered by:

low classifier confidence
weak or missing historical evidence
high-risk language
account/security-sensitive requests
explicit requests for human assistance

If automated generation fails, the system also falls back to:

HUMAN

rather than reporting a successful automated response.

7. Setup
Requirements

Development and evaluation were performed using Python 3.13.

Create a virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Environment Variables

Create a local .env file in the project root:

HF_TOKEN=your_hugging_face_token

The token is used by the response-generation component through the Hugging Face inference router.

Do not commit .env.

8. Running the Agent

Before running the full agent, ensure the processed historical dataset exists at:

data/processed/amazonhelp_direct_interactions.csv

Then run:

python src\pipeline.py

Example input:

My package says delivered but I never received it.

The agent displays:

predicted intent
classifier confidence
top historical evidence
escalation decision
escalation reason
automated reply when AUTO handling is allowed

For example, the message above was classified as:

delivery_issue

with a low confidence score and therefore routed to:

HUMAN

rather than being automatically handled.

9. Evaluation
9.1 Intent Classification

Run:

python evaluation\evaluate_tfidf.py

The supervised evaluation uses the 200-example golden set with a leave-one-out protocol, while the HF zero-shot baseline is evaluated directly on the same examples without training on the set.

Results
Model	Accuracy	Macro F1
Majority baseline	10.0%	1.82%
HF zero-shot classifier	32.5%	28.81%
TF-IDF + Logistic Regression	88.0%	87.92%

The supervised classifier substantially outperforms the zero-shot baseline on the same 200-example evaluation set.

The strongest supervised categories include:

order_status_tracking: 1.00 F1
product_seller_availability: 0.97 F1
order_changes_cancellations: 0.93 F1
product_device_support: 0.92 F1

The main weaker categories are:

delivery_issue: 0.77 F1
general_feedback_non_actionable: 0.79 F1

These categories contain more semantic overlap with neighboring operational intents.

9.2 Historical Retrieval

Run:

python evaluation\evaluate_retrieval.py

Results:

Metric	Result
Recall@1	32.5%†
Recall@3	46.5%†
Recall@5	53.0%†

The increase from Recall@1 to Recall@5 indicates that useful related cases are often present among the retrieved results even when they are not ranked first.

† Metric caveat: Recall@K uses agreement between the golden intent and the intent predicted for retrieved historical customer messages as a proxy for relevance. It is not human-annotated retrieval relevance.

9.3 Escalation

Run:

python evaluation\evaluate_escalation.py

Results:

Metric	Result
Human precision	81.25%‡
Human recall	92.86%‡
Human F1	86.67%‡
False auto-handle rate	7.14%‡

The policy is intentionally conservative and favors human review when confidence, evidence, or safety conditions are insufficient.

‡ Metric caveat: This is a policy-trigger benchmark using independently defined high-risk, security, and explicit-human-request rules. It is not a human-labelled production escalation dataset.

9.4 Reply-Quality Evaluation

A 50-example response-generation evaluation was attempted.

Metric	Result
Examples attempted	50
Successfully judged replies	1
Generation failures	45
Judge failures	4
Generation failure rate	90.0%

The single successfully judged response received:

Quality Dimension	Score
Groundedness	2.00 / 2.00
Correctness	2.00 / 2.00
Relevance	2.00 / 2.00
Completeness	2.00 / 2.00
Tone	2.00 / 2.00

This result is not used as an aggregate response-quality score because only one response was successfully judged.

The primary cause of failure was exhaustion of the included Hugging Face inference credits during evaluation.

10. Why the Supervised Classifier Was Chosen

The first prototype used a Hugging Face zero-shot classifier.

The fair evaluation showed:

HF zero-shot:
32.5% accuracy
28.81% macro F1

The supervised classifier achieved:

TF-IDF + Logistic Regression:
88.0% accuracy
87.92% macro F1

Because both approaches were evaluated on the same 200-example golden evaluation set, the supervised classifier's substantially higher performance motivated its selection as the primary intent-classification component.

The zero-shot model remains useful as a baseline because it demonstrates the limitations of generic zero-shot matching for this fine-grained taxonomy.

11. Safety Design

The agent follows a conservative handling policy.

Low Confidence

If classifier confidence is below:

0.45

the request is escalated to HUMAN.

Example:

Customer:
My package says delivered but I never received it.

Predicted intent:
delivery_issue

Confidence:
0.2278

Decision:
HUMAN

The classification is correct, but the confidence is insufficient for safe autonomous handling.

Weak Historical Evidence

If the retriever cannot provide useful historical evidence above the configured similarity threshold, the request is escalated.

High-Risk or Security-Sensitive Requests

High-risk, fraud-related, legal, safety, and account-security language can force HUMAN escalation regardless of classifier confidence.

Explicit Human Requests

Requests such as:

speak to a manager
speak to a human agent
real person
representative

are escalated directly.

12. Grounding Strategy

The response generator receives historical AmazonHelp examples containing:

historical customer message
historical AmazonHelp reply
retrieval similarity

The generator is explicitly instructed to avoid unsupported claims.

It must not invent:

company policies
refund decisions
delivery dates
account actions
compensation
guarantees
private account information

The objective is to make the response evidence-backed, even when that means asking for more information rather than pretending to know the answer.

13. Top Failure Modes
13.1 Fine-Grained Intent Confusion

The zero-shot baseline has difficulty separating adjacent operational intents.

For example, tracking, delivery, device, digital-content, and seller-related language can overlap even when the required resolution is different.

The supervised classifier improves this substantially but does not eliminate the problem.

Hypothesis: a hierarchical classifier or additional labelled examples around difficult boundaries would improve the remaining confusion.

13.2 Confidence Calibration

The classifier can make the correct prediction while assigning relatively low confidence.

Example:

delivery_issue
confidence = 0.2278

This makes raw confidence unsuitable as a calibrated probability without additional validation.

Hypothesis: calibration on a held-out validation set would make confidence-based automation thresholds more defensible.

13.3 Retrieval Ranking Limitations

Recall@1 is substantially lower than Recall@5.

This indicates that useful evidence can exist among the retrieved results without being the top-ranked historical example.

Hypothesis: combining lexical retrieval with semantic embeddings could improve ranking quality.

13.4 Low-Information Historical Replies

Some historical AmazonHelp replies are generic requests for additional information or redirects.

These responses provide limited resolution evidence.

Hypothesis: historical examples could be ranked or filtered based on how informative the resolution is, rather than similarity alone.

13.5 External Inference Reliability

The response-generation evaluation produced a 90% generation failure rate after included provider credits were exhausted.

This demonstrates that external inference availability is an operational dependency.

Hypothesis: a production system would need provider fallback, rate-limit monitoring, retries, caching, and graceful degradation.

14. What Is Misleading About the 88.0% Headline Number?

The 88.0% intent accuracy should not be interpreted as 88% production-safe autonomous customer support.

Several limitations affect that interpretation:

The evaluation set was human-verified, but it was not independently annotated by multiple human reviewers.
The 10-intent taxonomy contains closely related operational categories.
Retrieval Recall@K is based on an intent-agreement proxy rather than human relevance judgement.
Escalation is evaluated using policy-defined triggers rather than production-labelled decisions.
Automated reply-quality evaluation was constrained by external inference-provider credit exhaustion.

The results therefore demonstrate meaningful prototype-level engineering performance and reveal useful failure modes, but they do not establish production-level autonomous support quality.

15. Decision Log
1. Restrict the prototype to AmazonHelp

AmazonHelp provides a sufficiently large and internally consistent historical support corpus while avoiding cross-brand differences in policies and response styles.

2. Use a compact 10-intent taxonomy

A small operational taxonomy keeps classification tractable while covering the main support workflows represented in the selected data.

3. Define intents by primary customer need

Primary-need classification reduces ambiguity caused by overlapping keywords such as “Prime,” “package,” “Kindle,” and “seller.”

4. Keep escalation separate from intent

Intent determines what the customer needs; escalation determines whether the system can safely handle it.

5. Use TF-IDF + Logistic Regression as the primary classifier

It substantially outperformed the evaluated zero-shot baseline under the same evaluation protocol.

6. Retain zero-shot classification as a baseline

The baseline provides evidence about the limitations of generic zero-shot classification for this support taxonomy.

7. Use historical AmazonHelp conversations for retrieval

Historical resolutions provide brand-specific evidence for response generation.

8. Start with TF-IDF retrieval

It provides a simple, transparent, locally reproducible retrieval approach without unnecessary infrastructure.

9. Treat Recall@K as a proxy

Intent agreement provides an automated retrieval signal, but it is explicitly not treated as human relevance judgement.

10. Use conservative confidence-based escalation

Low-confidence classifications should not be treated as sufficiently trustworthy for autonomous handling.

11. Give safety rules precedence

High-risk, security-sensitive, and explicit human requests should override ordinary automation decisions.

12. Constrain generation to retrieved evidence

The generator should not invent policies, actions, dates, refunds, or guarantees unsupported by historical evidence.

13. Prefer safety over maximum automation

The system is intentionally designed to escalate uncertain cases rather than maximize the AUTO rate.

14. Do not report the single successful reply score as representative

A perfect score from one successfully judged response would be misleading after 49 unsuccessful evaluations.

15. Treat provider reliability as an engineering concern

External inference failures must result in graceful human fallback rather than an apparently successful autonomous response.

16. Next Improvements

The highest-value next steps are:

have a second human annotator independently review the golden evaluation set
calibrate classifier confidence
improve retrieval using hybrid lexical + semantic search
filter or weight historical cases by resolution usefulness
add inference-provider fallback and rerun response-quality evaluation
17. Limitations

This project is an evaluated prototype rather than a production support system.

It does not currently provide:

independently double-annotated golden evaluation data
human-labelled retrieval relevance
calibrated classifier probabilities
reliable large-scale automated response-quality metrics
production-grade multi-provider inference failover

These limitations are intentionally disclosed rather than hidden behind headline metrics.

18. Submission Artifacts

This repository contains:

runnable source code
evaluation scripts
evaluation outputs
human-verified golden evaluation set
reproducibility requirements
project documentation

A separate six-page report accompanies the repository and summarizes the problem framing, architecture, results, failure modes, limitations, and decision log.

19. Dataset and License Note

The underlying Customer Support on Twitter dataset is subject to its original dataset license and terms.

The repository does not redistribute the full raw dataset.

The processed historical AmazonHelp corpus is expected to be supplied locally when reproducing the full retrieval pipeline.