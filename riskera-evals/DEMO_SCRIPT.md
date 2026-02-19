# 🎯 VaultScan Evals — Walkthrough Guide

> This guide walks you through the VaultScan evaluation framework — what it does, how it works, and how to run it yourself.
>
> **Prerequisites:** Python venv activated, `.env` configured with Azure AI Foundry credentials

```powershell
# Activate environment before starting
.\venv\Scripts\Activate.ps1
```

---

## 1. 🔴 The Problem We're Solving

- 🌍 **Non-standard formats everywhere** — Swiss SME annual reports, German HGB filings, UK Companies House, emerging market statements — none follow US IFRS conventions. Every document is a snowflake.
- 📉 **No systematic accuracy measurement** — Today, there's no way to answer "how accurate is our extraction pipeline?" with a number. Quality is anecdotal, not quantified.
- ⏰ **No degradation detection** — When Azure Doc Intelligence updates, when prompts change, when new document formats appear — we have no early warning system. Problems surface only when SMEs catch them manually.
- 💸 **SME time is expensive** — Credit analysts spend hours reviewing extractions that may already be correct, because there's no confidence signal telling them what actually needs attention.

> 💡 What if you could measure accuracy per format, detect degradation automatically, and only route uncertain extractions to SMEs? That's exactly what this framework does.

---

## 2. 📐 The Methodology: Dual Evaluation Strategy

### Offline Evaluation (Pre-production)
- 🧪 **A/B test prompt strategies** — Compare two extraction configurations with statistical significance testing
- 🔁 **Regression testing** — Run against a golden dataset on every pipeline change; fail CI if accuracy drops below threshold
- 📊 **Per-format accuracy breakdown** — Know exactly which document formats are hardest (Swiss SME vs. German HGB vs. UK)

### Online Evaluation (Production)
- 📡 **Real-time monitoring** — Every extraction scored instantly, streamed to Application Insights
- 🎯 **Confidence-based SME routing** — Only route documents below confidence threshold to human review (reduce SME load by 40-60%)
- 📉 **Drift detection** — Rolling window tracks accuracy over last 100 extractions; alerts on 15% degradation

### Five Custom Evaluators
| Evaluator | What it measures | Weight | Ground Truth Source |
|---|---|---|---|
| **Numerical Accuracy** | % deviation from SME-validated values (0-5 scale) | 30% | 🏷️ SME-labeled golden dataset (offline) |
| **Metric Completeness** | Were all 22 required metrics extracted? | 20% | 📋 Required metrics list |
| **Math Consistency** | Do extracted values satisfy financial equations? (e.g. gross_profit = revenue − COGS) | 20% | 📐 Accounting principles |
| **YoY Consistency** | Do year-over-year changes make financial sense? | 15% | 📅 Prior year actuals |
| **Groundedness** | Are extracted values traceable to source document? | 15% | 📄 Source document text |

> 💡 **Key insight:** _We have **multiple layers of ground truth** in production — the source document for grounding, accounting rules for math consistency, prior year data for trend validation. The only thing missing in production is SME-validated labels — which is what the offline golden dataset provides. The feedback loop closes the gap: low-confidence extractions get SME review, corrections become golden dataset entries, and the system gets smarter over time._

> Let's see this running against synthetic data that mirrors your actual document formats.

---

## 3. 🖥️ Try It Yourself

### Step 1: Review Configuration (30 sec)

```powershell
python run_evaluation.py config
```

**What you'll see:**
- ✅ Azure AI Foundry connection, model deployment, and evaluation thresholds
- ✅ Numerical deviation bands: <0.1% = perfect, <1% = excellent, <5% = good, <10% = acceptable
- ✅ YoY anomaly thresholds: >50% revenue swing, >30% asset swing, sign flips

---

### Step 2: Preview Document Formats (1 min)

```powershell
python run_evaluation.py generate --preview-all-formats
```

**What you'll see:**
- 📄 **5 format templates** that mirror real-world client documents:
  - 🇨🇭 Swiss SME — German-language labels, CHF, unique formatting
  - 🇩🇪 German HGB — Handelsgesetzbuch conventions, EUR
  - 🇬🇧 UK Companies House — Abbreviated accounts, GBP
  - 📝 Unstructured narrative — Metrics buried in prose paragraphs
  - 📊 Table-based — Clean tabular layout (the "easy" format)
- 🎯 Each template has a difficulty rating — this is why per-format accuracy matters

---

### Step 3: Generate Golden Dataset (1 min)

```powershell
python run_evaluation.py generate --size 20 --output data/golden_dataset.json
```

**What you'll see:**
- 🏗️ Generates 20 synthetic documents with **known ground truth** values
- 💰 Financially consistent: revenue → COGS → gross profit → EBITDA → net income all follow realistic ratios
- 📅 Includes prior-year metrics with ~10% YoY variation for consistency checking
- 🎯 Mix of difficulty levels (easy/medium/hard) across all 5 format types
- 📋 This is what a real golden dataset looks like — yours would use SME-validated real extractions

---

### Step 4: Online Evaluation — Single Document (3 min)

```powershell
python run_evaluation.py online --demo --no-telemetry
```

**What you'll see:**
- 🔍 Extracts all **22 financial metrics** from a single document in real-time
- 📊 Each metric gets: extracted value, confidence score, source location in document
- 🧮 **Five evaluator scores** computed instantly:
  - Numerical Accuracy: how close to ground truth? _(offline only — needs SME-validated data)_
  - Completeness: any missing metrics?
  - **Math Consistency: do the numbers add up?** _(this is the key production signal — checks 9 financial equations like gross_profit = revenue − COGS, balance sheet identity, ratio calculations)_
  - YoY Consistency: any suspicious year-over-year changes?
  - Groundedness: can we trace values back to the source?
- 🚦 **Routing decision**: HIGH confidence → auto-approve | LOW confidence → route to SME queue
- 💡 **Multiple layers of ground truth in production** — the source document itself (grounding), accounting principles (math consistency), prior year actuals (YoY). The only layer missing live is SME-validated labels — which the feedback loop continuously builds.
- ⚡ This runs on every extraction in production. SMEs only see what actually needs review.

> **FAQ: How does this work without ground truth?** Ground truth is derived from multiple sources. The source document is ground truth for grounding checks — if the model says revenue is 15M, we verify that "15,000,000" appears in the document. Accounting principles are ground truth for math consistency — gross profit must equal revenue minus COGS, the balance sheet must balance. Prior year actuals are ground truth for trend validation. The only thing not available in production is SME-validated "correct answers" — and that's exactly what the feedback loop builds. Every SME correction on a routed extraction becomes a new golden dataset entry.

---

### Step 5: Offline Batch Evaluation (4 min)

```powershell
python run_evaluation.py offline --dataset data/golden_dataset.json --no-groundedness
```

> 💡 We skip groundedness here for speed — it requires an additional LLM call per document.

**What you'll see:**
- 📊 **Per-format accuracy breakdown** — Notice the difference between table-based (easy) and unstructured narrative (hard). This tells you exactly where to invest in prompt engineering.
- 📋 **Per-metric accuracy** — Some metrics extract reliably (revenue, total assets), others are consistently harder (working capital, interest coverage ratio)
- ❌ **Completeness gaps** — Which metrics are most frequently missed, and in which format types?
- 🎯 **Aggregate score** — Single number (0-5 scale) to track over time; CI/CD threshold is 3.5

**If time permits — A/B comparison:**

```powershell
python run_evaluation.py compare --config-a "baseline" --config-b "enhanced_prompts" --dataset data/golden_dataset.json
```

- 📈 Side-by-side accuracy comparison with statistical significance testing
- This is how you can prove that a prompt change actually improves extraction — not just on one document, but across all formats.

---

## 4. 💡 Key Benefits & Value Proposition

| Capability | Business Value |
|---|---|
| 📊 **Quantified accuracy per format** | Target prompt improvements where they matter most — don't waste effort on formats that already work |
| 🧮 **Multi-layered ground truth** | Every evaluator has its own ground truth source — source document, accounting rules, prior year actuals. No cold-start problem; works from day one |
| 🚨 **Automated regression detection** | Catch degradation from model updates, prompt changes, or new document types _before_ it reaches production |
| 🎯 **Confidence-based SME routing** | Only route what needs human review — reduce SME review load by 40-60% while maintaining quality |
| 🔄 **Continuous improvement loop** | SME corrections on routed documents feed back into the golden dataset — the system gets smarter over time. Derived ground truth today → labeled ground truth tomorrow |
| ✅ **CI/CD quality gates** | No pipeline change ships if accuracy drops below threshold — quality is enforced, not hoped for |

> **Bottom line:** This isn't just testing — it's a quality infrastructure layer that sits on top of your existing pipeline. Azure Doc Intelligence → LLM extraction → **evaluation & routing** → SME review. Every component you already have stays. We're adding the measurement and feedback loop. Each evaluator derives ground truth from a different source — the source document, accounting principles, prior year actuals — so you get immediate value from day one.

---

## 5. 🚀 Recommended Next Steps

### Immediate (Weeks 1-2)
- 📋 **Collect 100-200 real SME-validated extractions** across format types to build the production golden dataset
- 📏 **Baseline current pipeline accuracy** — run offline evaluation against real data to get the starting numbers

### Short-term (Weeks 3-4)
- 🧪 **A/B test prompt strategies** with statistical significance across Swiss SME, German HGB, UK formats
- 🔧 **Tune confidence thresholds** — calibrate routing so SMEs get the right volume of reviews

### Medium-term (Month 2)
- 📡 **Deploy online monitoring** with Application Insights dashboards and alerting
- 🔁 **Set up CI/CD quality gates** — automated evaluation on every prompt or pipeline change
- 📊 **Build format-specific prompt strategies** — use per-format accuracy data to optimize where it matters

### Ongoing
- 🔄 **Feedback loop** — SME corrections automatically enrich the golden dataset
- 📈 **Track accuracy trends** over time — demonstrate measurable improvement to stakeholders

> **The bottom line:** The question isn't whether extraction accuracy matters — it's whether you can measure it. This framework gives you the numbers, the alerts, and the feedback loop to continuously improve.

---

## 📎 Appendix: Quick Reference Commands

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Show configuration and validate setup
python run_evaluation.py config

# Preview document format templates
python run_evaluation.py generate --preview-all-formats

# Generate golden dataset (20 documents)
python run_evaluation.py generate --size 20 --output data/golden_dataset.json

# Online evaluation — single document demo
python run_evaluation.py online --demo --no-telemetry

# Offline batch evaluation
python run_evaluation.py offline --dataset data/golden_dataset.json --no-groundedness

# Offline evaluation with groundedness (slower, requires LLM)
python run_evaluation.py offline --dataset data/golden_dataset.json

# A/B comparison of two configurations
python run_evaluation.py compare --config-a "baseline" --config-b "enhanced_prompts" --dataset data/golden_dataset.json

# Offline evaluation in CI mode (exits non-zero if below threshold)
python run_evaluation.py offline --dataset data/golden_dataset.json --ci --threshold 3.5
```
