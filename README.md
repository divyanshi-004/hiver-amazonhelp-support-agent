# Evaluation Results

> **Evaluation focus:** intent classification, historical retrieval, safe escalation, and grounded response generation.

---

## 1. Headline Results

| System Component | Metric | Result |
|:---|:---|---:|
| Majority baseline | Accuracy | **10.0%** |
| HF zero-shot classifier | Accuracy | **32.5%** |
| HF zero-shot classifier | Macro F1 | **28.81%** |
| TF-IDF + Logistic Regression | Accuracy | **88.0%** |
| TF-IDF + Logistic Regression | Macro F1 | **87.92%** |
| Historical retrieval | Recall@1 | **32.5%†** |
| Historical retrieval | Recall@3 | **46.5%†** |
| Historical retrieval | Recall@5 | **53.0%†** |
| Escalation policy | Human F1 | **86.67%‡** |
| Escalation policy | Human recall | **92.86%‡** |
| Escalation policy | Human precision | **81.25%‡** |
| Escalation policy | False auto-handle rate | **7.14%‡** |
| Reply generation | Generation failure rate | **90.0%§** |

The intent-classification results use the same 200-example evaluation set and evaluation protocol for both classifiers.

---

## 2. What the Results Mean

### Intent Classification

The supervised **TF-IDF + Logistic Regression classifier** substantially outperforms the HF zero-shot classifier on the same 200-example evaluation set:

| Model | Accuracy | Macro F1 |
|:---|---:|---:|
| Majority baseline | 10.0% | 1.82% |
| HF zero-shot | 32.5% | 28.81% |
| **TF-IDF + Logistic Regression** | **88.0%** | **87.92%** |

This result motivated the decision to use the supervised classifier as the primary intent-classification component and retain the HF zero-shot model as a baseline.

The strongest supervised categories include:

- `order_status_tracking`: **1.00 F1**
- `product_seller_availability`: **0.97 F1**
- `order_changes_cancellations`: **0.93 F1**
- `product_device_support`: **0.92 F1**

The main remaining weaknesses are `delivery_issue` (**0.77 F1**) and `general_feedback_non_actionable` (**0.79 F1**), where messages can overlap semantically with neighbouring operational categories.

The HF zero-shot classifier performs substantially worse, with **0% recall** for both `order_status_tracking` and `product_seller_availability`. This demonstrates that zero-shot label matching is not sufficiently reliable for the project's fine-grained intent taxonomy.

### Historical Retrieval

The retrieval system achieves:

- **32.5% Recall@1**
- **46.5% Recall@3**
- **53.0% Recall@5**

The improvement from top-1 to top-5 indicates that useful historical cases are often present somewhere in the retrieved evidence, even when they are not ranked first.

> **Metric caveat:** Recall@K uses agreement between the golden intent and the intent predicted for retrieved historical customer messages as a proxy for relevance. It is **not human-annotated retrieval relevance**.

### Escalation Safety

The escalation layer achieves:

- **86.67% Human-escalation F1**
- **92.86% Human-escalation recall**
- **81.25% Human-escalation precision**
- **7.14% false auto-handle rate**

The system deliberately prefers **safe escalation over unsupported automation**. For example, low classifier confidence, security-related language, high-risk terms, and explicit requests for a human trigger escalation.

This benchmark uses independently defined triggers for **high-risk issues, account-security issues, and explicit requests for a human**. It is not a production human-labelled escalation dataset.

---

## 3. Reply-Quality Evaluation

The reply-quality evaluation attempted **50 examples**.

| Metric | Result |
|:---|---:|
| Examples attempted | 50 |
| Successfully judged replies | 1 |
| Generation failures | 45 |
| Judge failures | 4 |
| Generation failure rate | **90.0%** |

The single successfully judged reply received:

| Quality Dimension | Score |
|:---|---:|
| Groundedness | 2.00 / 2.00 |
| Correctness | 2.00 / 2.00 |
| Relevance | 2.00 / 2.00 |
| Completeness | 2.00 / 2.00 |
| Tone | 2.00 / 2.00 |

These scores are **not treated as a meaningful aggregate quality estimate**, because only one generated reply was successfully judged.

The primary failure was exhaustion of the included Hugging Face inference credits during evaluation.

> **Decision:** No aggregate automated reply-quality score is claimed from this run.

The failure itself is an important engineering finding: the agent's response-generation component currently depends on an external inference provider and needs graceful degradation when that provider becomes unavailable.

---

## 4. What Is Misleading About My Headline Number?

The largest number in the evaluation is **88.0% classifier accuracy**, but it should not be interpreted as production-ready autonomous customer-support performance.

Several factors limit the conclusions that can be drawn:

1. The evaluation set is **curated rather than independently human-annotated**.

2. The intent taxonomy contains **closely related operational categories**, so classification accuracy depends strongly on the quality and consistency of the taxonomy boundaries.

3. Retrieval relevance is measured using an **intent-agreement proxy**, not human relevance judgements.

4. The escalation metrics come from a **policy-trigger benchmark**, not human-labelled production decisions.

5. Reply-quality evaluation was severely constrained by **external inference-provider credit exhaustion**, so no reliable aggregate reply-quality score is reported.

Therefore, the results demonstrate strong prototype-level classification performance and useful safety signals, but **they do not establish production-level autonomous customer-support quality**.

---

## 5. Top Failure Modes

### 5.1 Fine-Grained Intent Confusion

The HF zero-shot classifier records **0% recall** for both `order_status_tracking` and `product_seller_availability`.

The supervised classifier improves substantially, but some confusion remains between semantically adjacent categories such as delivery, tracking, general feedback, and account-related issues.

**Example:** messages about missing or delayed orders can contain similar vocabulary even when their intended operational resolution differs.

**Hypothesis:** a supervised classifier trained on the project-specific taxonomy provides cleaner boundaries than zero-shot semantic matching, but a hierarchical classifier or additional training data could further improve difficult category boundaries.

---

### 5.2 Weak Confidence Calibration

The supervised classifier can correctly rank an intent while assigning relatively low probability to it.

For example:

> `My package says delivered but I never received it.`

was classified as:

> `delivery_issue` — **0.2278**

The prediction is correct, but the confidence is low enough to trigger HUMAN escalation under the current safety policy.

**Hypothesis:** the model's probabilities should be calibrated on a validation set before being interpreted as estimates of correctness. This would make confidence-based escalation thresholds more defensible.

---

### 5.3 Retrieval Ranking Limitations

Retrieval reaches **32.5% Recall@1** and **53.0% Recall@5**.

This suggests that useful historical evidence is frequently present among the top retrieved cases, but the best case is not always ranked first.

**Hypothesis:** TF-IDF captures lexical similarity effectively but does not always capture whether two messages require the same operational resolution. Hybrid lexical + semantic retrieval could improve ranking.

---

### 5.4 Low-Information Historical Responses

Many historical AmazonHelp responses are generic requests for more information or redirects to another support channel.

This limits how much useful resolution information can be extracted from the historical corpus.

**Hypothesis:** weighting or filtering historical cases by resolution usefulness could provide stronger evidence to the response generator.

---

### 5.5 External Inference Reliability

The reply evaluation experienced a **90% generation failure rate** after the included Hugging Face inference credits were exhausted.

This means the prototype currently has an operational dependency outside the core application logic.

**Hypothesis:** a production system should provide provider fallback, caching, retry handling, rate-limit monitoring, and graceful degradation when inference becomes unavailable.

---

## 6. Evaluation Limitations

The current results should be interpreted as an **engineering evaluation of a prototype**, not as a production certification.

The most important next improvements are:

- create a genuinely **human-verified golden set**
- add **human-labelled retrieval relevance**
- evaluate replies with a stable inference provider and a larger judged sample
- calibrate classifier confidence before using it for automation decisions
- improve grounding so generated replies remain strictly within the retrieved historical evidence

---

### Notes

**†** Retrieval Recall@K uses intent agreement as a proxy for historical-case relevance and is not human-annotated retrieval relevance.

**‡** Escalation metrics come from an independently defined safety-trigger benchmark rather than a human-labelled production escalation dataset.

**§** The reply-quality run produced 45 generation failures and 4 judge failures out of 50 attempts. The resulting 100% quality score from the single judged reply is therefore **not a meaningful estimate of reply quality**.