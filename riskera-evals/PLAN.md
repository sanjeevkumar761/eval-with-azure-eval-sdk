# Financial Metrics Extraction Evaluation Framework

## Executive Summary

This framework provides a comprehensive evaluation system for measuring and improving the accuracy of financial metrics extraction from non-standardized financial statements issued by SME corporate clients. It addresses the challenge of extracting accurate financial data from documents that do not follow standard formats (unlike US IFRS-aligned statements).

## Problem Statement

- **Challenge**: Non-standardized financial statements from SME clients across various markets
- **Current Approach**: Azure Doc Intelligence → LLM with bespoke prompts → SME augmentation → Mathematical solvers
- **Gap**: No systematic way to measure extraction accuracy or detect degradation over time
- **Goal**: Build an evaluation framework using Azure AI Foundry and Azure AI Evaluation SDK

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE EVALUATION                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Golden       │───▶│ Extraction   │───▶│ Evaluators   │───▶│ Reports &    │   │
│  │ Dataset      │    │ Agent        │    │ (Batch)      │    │ Metrics      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                         ONLINE EVALUATION                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Production   │───▶│ Extraction   │───▶│ Evaluators   │───▶│ App Insights │   │
│  │ Documents    │    │ Agent        │    │ (Per-request)│    │ + Alerts     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                                       │                               │
│         ▼                                       ▼                               │
│  ┌──────────────┐                      ┌──────────────┐                        │
│  │ SME Review   │◀─────────────────────│ Low-Confidence│                        │
│  │ Queue        │  (flagged samples)   │ Detection     │                        │
│  └──────────────┘                      └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dual Evaluation Strategy

| Mode | Purpose | When | Data Source |
|------|---------|------|-------------|
| **Offline** | A/B testing, regression testing, prompt optimization | Pre-deployment | Golden dataset (SME-validated) |
| **Online** | Production monitoring, drift detection, alerting | Runtime | Live extractions + sampling |

---

## File Structure

```
vaultscan-evals/
├── PLAN.md                    # This documentation
├── requirements.txt           # Python dependencies
├── .env.template              # Environment variables template
│
├── config.py                  # Configuration & environment setup
├── models.py                  # Pydantic data models for financial metrics
├── custom_evaluators.py       # Custom evaluators (Numerical Accuracy, Completeness, Consistency)
├── extraction_agent.py        # Azure AI Foundry agent for metric extraction
│
├── offline_evaluation.py      # Batch evaluation against golden dataset
├── online_evaluation.py       # Production middleware with real-time eval
├── golden_dataset.py          # Sample golden dataset generator/loader
└── run_evaluation.py          # CLI entry point for both modes
```

---

## Custom Evaluators

### 1. NumericalAccuracyEvaluator

Measures the percentage deviation between extracted numerical values and ground truth.

| Input | Output |
|-------|--------|
| `extracted_value`, `ground_truth_value` | Score (0-5), deviation percentage, pass/fail |

**Scoring Logic:**
- Score 5: Deviation < 0.1%
- Score 4: Deviation < 1%
- Score 3: Deviation < 5%
- Score 2: Deviation < 10%
- Score 1: Deviation < 25%
- Score 0: Deviation >= 25%

### 2. MetricCompletenessEvaluator

Measures what percentage of required financial metrics were successfully extracted.

| Input | Output |
|-------|--------|
| `extracted_metrics`, `required_metrics` | Score (0-5), completeness %, missing list |

**Scoring Logic:**
- Score 5: 100% complete
- Score 4: >= 90% complete
- Score 3: >= 75% complete
- Score 2: >= 50% complete
- Score 1: >= 25% complete
- Score 0: < 25% complete

### 3. YoYConsistencyEvaluator

Checks if year-over-year changes are within reasonable bounds (flags anomalies).

| Input | Output |
|-------|--------|
| `current_metrics`, `prior_year_metrics`, `thresholds` | Score (0-5), anomaly list, explanations |

**Anomaly Detection:**
- Revenue change > 50% YoY → Flag
- Net income sign flip → Flag
- Asset/Liability ratio swing > 30% → Flag

### 4. GroundednessEvaluator (Built-in SDK)

Measures if extracted values are grounded in the source document.

| Input | Output |
|-------|--------|
| `query`, `context`, `response` | Score (0-5), reason, pass/fail |

---

## Required Financial Metrics

```python
REQUIRED_METRICS = [
    # Income Statement
    "total_revenue",
    "cost_of_goods_sold",
    "gross_profit",
    "operating_expenses",
    "operating_income",
    "interest_expense",
    "net_income",
    "ebitda",
    
    # Balance Sheet
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "shareholders_equity",
    "retained_earnings",
    
    # Cash Flow
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    
    # Ratios
    "current_ratio",
    "debt_to_equity",
    "gross_margin",
    "net_margin",
    "return_on_equity",
]
```

---

## Offline Evaluation

### Features

- **Batch Processing**: Evaluate entire golden dataset in parallel
- **Per-Metric Breakdown**: Accuracy statistics per metric type
- **A/B Testing**: Compare different prompting strategies with statistical significance
- **Regression Testing**: Detect accuracy degradation when models/prompts change
- **Export**: JSON/CSV reports for analysis
- **CI/CD Integration**: Exit codes and JUnit-compatible output

### Usage

```bash
# Run offline evaluation
python run_evaluation.py offline --dataset golden_dataset.json --output results/

# A/B test two prompt versions
python run_evaluation.py offline --compare prompt_v1.txt prompt_v2.txt

# CI/CD mode (returns exit code 1 if below threshold)
python run_evaluation.py offline --ci --threshold 3.5
```

### Output Example

```json
{
  "evaluation_id": "eval_2026_02_18_001",
  "timestamp": "2026-02-18T14:30:00Z",
  "dataset_size": 150,
  "overall_scores": {
    "numerical_accuracy": 4.2,
    "completeness": 4.5,
    "yoy_consistency": 3.8,
    "groundedness": 4.1
  },
  "per_metric_accuracy": {
    "total_revenue": {"mean": 4.5, "std": 0.3},
    "net_income": {"mean": 3.9, "std": 0.7},
    "ebitda": {"mean": 3.2, "std": 1.1}
  },
  "failed_extractions": [
    {"document_id": "client_042", "metric": "operating_cash_flow", "reason": "Not found in document"}
  ]
}
```

---

## Online Evaluation

### Features

- **Per-Request Evaluation**: < 500ms overhead per extraction
- **Confidence Scoring**: Aggregate confidence from all evaluators
- **Low-Confidence Routing**: Flag extractions for SME review
- **Real-Time Telemetry**: Metrics to Azure Application Insights
- **Drift Detection**: Rolling accuracy windows with alerts
- **Alert Triggers**: Automatic notifications on accuracy degradation

### Application Insights Metrics

| Metric Name | Type | Alert Threshold |
|-------------|------|-----------------|
| `extraction_accuracy_score` | Gauge | < 3.5 (out of 5) |
| `metric_completeness_rate` | Percentage | < 85% |
| `numerical_deviation_avg` | Percentage | > 5% |
| `low_confidence_rate` | Percentage | > 15% |
| `yoy_anomaly_count` | Counter | > 3 per document |
| `extraction_latency_ms` | Histogram | p95 > 5000ms |

### Usage

```python
from online_evaluation import OnlineEvaluationMiddleware

# Initialize middleware
middleware = OnlineEvaluationMiddleware(
    confidence_threshold=0.7,
    enable_app_insights=True,
    sme_queue_connection="..."
)

# Wrap extraction calls
result = await middleware.extract_with_evaluation(
    document_text=doc_text,
    prior_year_metrics=prior_metrics,
)

# Result includes extraction + evaluation
print(result.extracted_metrics)
print(result.confidence_score)
print(result.evaluation_details)
print(result.flagged_for_review)  # True if confidence < threshold
```

### SME Feedback Loop

```
Production Extraction
        │
        ▼
┌───────────────────┐
│ Confidence < 0.7? │──Yes──▶ Queue for SME Review
└───────────┬───────┘                 │
            │ No                      ▼
            ▼                 ┌───────────────┐
     Use extraction           │ SME Corrects  │
                              └───────┬───────┘
                                      │
                                      ▼
                              Add to Golden Dataset
                              (continuous improvement)
```

---

## Configuration

### Environment Variables

```bash
# Azure AI Foundry
AZURE_AI_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api/projects/<id>
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-4o

# Application Insights (Online Evaluation)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# Evaluation Thresholds
NUMERICAL_ACCURACY_THRESHOLD=3.5
COMPLETENESS_THRESHOLD=4.0
CONSISTENCY_THRESHOLD=3.0
CONFIDENCE_THRESHOLD=0.7

# SME Review Queue (Azure Service Bus)
SME_QUEUE_CONNECTION_STRING=Endpoint=sb://...
SME_QUEUE_NAME=sme-review-queue
```

---

## Integration with Existing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Current Pipeline                                    │
│                                                                             │
│  Financial    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  Statement ──▶│ Azure Doc    │───▶│ LLM + SME    │───▶│ Math Solver  │──┐  │
│               │ Intelligence │    │ Prompts      │    │              │  │  │
│               └──────────────┘    └──────────────┘    └──────────────┘  │  │
│                                                                          │  │
│  ┌──────────────────────────────────────────────────────────────────────┼──┤
│  │                    + Evaluation Layer                                 │  │
│  │                                                                       ▼  │
│  │  Prior Year ──▶ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Metrics        │ Online Eval  │───▶│ Confidence   │───▶│ Final Output │ │
│  │                 │ Middleware   │    │ Scoring      │    │ or SME Queue │ │
│  │                 └──────────────┘    └──────────────┘    └──────────────┘ │
│  └──────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```
azure-ai-evaluation>=1.0.0
azure-ai-projects>=1.0.0
azure-identity>=1.15.0
azure-monitor-opentelemetry>=1.0.0
azure-servicebus>=7.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
click>=8.0.0
```

---

## Getting Started

### 1. Setup Environment

```bash
cd vaultscan-evals
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your Azure credentials
```

### 2. Run Offline Evaluation

```bash
# Generate sample golden dataset
python golden_dataset.py --generate --size 50

# Run evaluation
python run_evaluation.py offline --dataset data/golden_dataset.json
```

### 3. Integrate Online Evaluation

```python
from online_evaluation import OnlineEvaluationMiddleware

middleware = OnlineEvaluationMiddleware()
result = await middleware.extract_with_evaluation(document_text)
```

---

## Success Metrics

| Metric | Current Baseline | Target |
|--------|------------------|--------|
| Numerical Accuracy Score | TBD | >= 4.0 |
| Metric Completeness | TBD | >= 90% |
| SME Review Rate | TBD | <= 15% |
| False Positive Rate | TBD | <= 5% |

---

## Next Steps

1. **Collect Golden Dataset**: Work with SMEs to validate 100-200 sample extractions
2. **Baseline Measurement**: Run offline evaluation on current pipeline
3. **Iterate on Prompts**: Use A/B testing to optimize extraction prompts
4. **Deploy Online Eval**: Integrate middleware into production pipeline
5. **Monitor & Alert**: Set up Application Insights dashboards and alerts
