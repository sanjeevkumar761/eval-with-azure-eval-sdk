# Data Directory

This directory contains:

- `golden_dataset.json` - SME-validated ground truth dataset for offline evaluation

## Generating a Dataset

```bash
python run_evaluation.py generate --size 50 --output data/golden_dataset.json
```

## Dataset Structure

Each entry contains:
- `document_id` - Unique identifier
- `raw_text` - Financial statement text (simulated)
- `ground_truth` - SME-validated metric values
- `prior_year` - Prior year metrics for YoY comparison
