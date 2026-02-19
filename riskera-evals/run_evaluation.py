"""
Financial Metrics Extraction Evaluation - CLI Entry Point
==========================================================

Unified command-line interface for running both offline and
online evaluations.

Usage:
    # Generate golden dataset
    python run_evaluation.py generate --size 50 --output data/golden_dataset.json
    
    # Run offline evaluation
    python run_evaluation.py offline --dataset data/golden_dataset.json
    
    # Run offline evaluation in CI mode
    python run_evaluation.py offline --dataset data/golden_dataset.json --ci --threshold 3.5
    
    # Run online evaluation demo
    python run_evaluation.py online --demo
    
    # A/B test comparison
    python run_evaluation.py compare --dataset data/golden_dataset.json --config-a prompt_v1.json --config-b prompt_v2.json
"""

import asyncio
import sys
import json
from pathlib import Path

import click

from config import get_config


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Financial Metrics Extraction Evaluation Framework.
    
    Evaluate accuracy of financial metrics extraction from
    non-standardized corporate financial statements.
    """
    pass


# =============================================================================
# Generate Command
# =============================================================================

@cli.command()
@click.option("--size", default=50, help="Number of entries to generate")
@click.option("--name", default="evaluation_dataset_v1", help="Dataset name")
@click.option("--output", default="data/golden_dataset.json", help="Output file path")
@click.option("--preview", is_flag=True, help="Preview a sample entry without saving")
@click.option("--preview-all-formats", is_flag=True, help="Preview all document format types")
def generate(size: int, name: str, output: str, preview: bool, preview_all_formats: bool):
    """Generate a synthetic golden dataset for evaluation."""
    from golden_dataset import generate_golden_dataset, generate_sample_document
    
    if preview_all_formats:
        format_names = [
            "Swiss SME (German language)",
            "German GmbH (Geschäftsbericht)",
            "UK Ltd (£ thousands)",
            "Narrative/Letter format",
            "Table-based with prior year",
        ]
        for idx, fmt_name in enumerate(format_names):
            click.echo(f"\n{'='*60}")
            click.echo(f"FORMAT {idx+1}: {fmt_name}")
            click.echo(f"{'='*60}")
            doc, metrics, _ = generate_sample_document(
                company_name=f"Demo Company {idx+1}",
                fiscal_year=2025,
                template_idx=idx,
            )
            click.echo(doc[:600] + "\n  ..." if len(doc) > 600 else doc)
        click.echo(f"\n{'='*60}")
        click.echo("These represent the variety of non-standardized formats")
        click.echo("that SME corporate clients issue across different markets.")
        click.echo(f"{'='*60}")
        return
    
    if preview:
        doc, metrics, prior = generate_sample_document(
            company_name="Sample Company GmbH",
            fiscal_year=2025,
        )
        click.echo("=== Sample Document ===")
        click.echo(doc)
        click.echo("\n=== Ground Truth Metrics ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                click.echo(f"  {k}: {v:,.2f}")
        return
    
    click.echo(f"Generating dataset with {size} entries...")
    dataset = generate_golden_dataset(
        name=name,
        size=size,
        output_path=output,
    )
    click.echo(f"Generated {dataset.size} entries")
    click.echo(f"Saved to: {output}")


# =============================================================================
# Offline Evaluation Command
# =============================================================================

@cli.command()
@click.option("--dataset", required=True, help="Path to golden dataset JSON file")
@click.option("--output", default="results/", help="Output directory for results")
@click.option("--ci", is_flag=True, help="CI mode - exit with code 1 if below threshold")
@click.option("--threshold", default=3.5, help="Minimum average score to pass (CI mode)")
@click.option("--format", "output_format", default="json", 
              type=click.Choice(["json", "csv", "both"]), help="Output format")
@click.option("--max-concurrent", default=5, help="Maximum concurrent evaluations")
@click.option("--no-groundedness", is_flag=True, help="Skip groundedness evaluation")
@click.option("--no-consistency", is_flag=True, help="Skip YoY consistency evaluation")
def offline(
    dataset: str,
    output: str,
    ci: bool,
    threshold: float,
    output_format: str,
    max_concurrent: int,
    no_groundedness: bool,
    no_consistency: bool,
):
    """Run offline evaluation against a golden dataset."""
    from offline_evaluation import OfflineEvaluator, OfflineEvaluationConfig
    
    # Validate dataset exists
    if not Path(dataset).exists():
        click.echo(f"Error: Dataset file not found: {dataset}", err=True)
        click.echo("Run 'python run_evaluation.py generate' to create a dataset first.", err=True)
        sys.exit(1)
    
    # Configure evaluation
    config = OfflineEvaluationConfig(
        max_concurrent=max_concurrent,
        include_groundedness=not no_groundedness,
        include_consistency=not no_consistency,
        output_format=output_format,
        fail_threshold=threshold,
    )
    
    async def run():
        evaluator = OfflineEvaluator(config=config)
        results = await evaluator.run(
            dataset_path=dataset,
            output_path=output,
        )
        
        # Load dataset for per-format analysis
        from golden_dataset import load_golden_dataset
        ds = load_golden_dataset(dataset)
        doc_metadata = {e.document_id: e for e in ds.entries}
        
        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo("EVALUATION SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Dataset: {results.dataset_name}")
        click.echo(f"Documents evaluated: {results.dataset_size}")
        click.echo(f"Duration: {results.total_duration_seconds:.1f}s")
        click.echo()
        click.echo("Overall Scores:")
        
        import statistics
        all_scores = []
        for evaluator_name, score in results.overall_scores.items():
            status = "✓" if score >= threshold else "✗"
            click.echo(f"  {status} {evaluator_name}: {score:.2f}")
            all_scores.append(score)
        
        avg_score = statistics.mean(all_scores) if all_scores else 0
        click.echo()
        click.echo(f"Average Score: {avg_score:.2f}")
        click.echo(f"Low confidence (flagged): {results.low_confidence_count}/{results.dataset_size}")
        click.echo(f"Failed extractions: {len(results.failed_extractions)}")
        
        # Per-format accuracy breakdown
        format_labels = {
            "template_0": "Swiss SME (German)",
            "template_1": "German GmbH",
            "template_2": "UK Ltd (£'000)",
            "template_3": "Narrative/Letter",
            "template_4": "Table-based",
        }
        format_scores = {}
        for doc_result in results.document_results:
            meta = doc_metadata.get(doc_result.document_id)
            if meta:
                fmt = meta.document_format or "unknown"
                label = format_labels.get(fmt, fmt)
                difficulty = meta.difficulty_level or "unknown"
                if label not in format_scores:
                    format_scores[label] = {"accuracy": [], "completeness": [], "difficulty": difficulty}
                format_scores[label]["accuracy"].append(doc_result.numerical_accuracy_score.score)
                format_scores[label]["completeness"].append(doc_result.completeness_score.score)
        
        if format_scores:
            click.echo()
            click.echo("-" * 60)
            click.echo("ACCURACY BY DOCUMENT FORMAT")
            click.echo("-" * 60)
            click.echo(f"  {'Format':<25} {'Difficulty':<12} {'Accuracy':>10} {'Completeness':>14}")
            click.echo(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*14}")
            for label, data in sorted(format_scores.items(), key=lambda x: statistics.mean(x[1]["accuracy"])):
                acc = statistics.mean(data["accuracy"])
                comp = statistics.mean(data["completeness"])
                diff = data["difficulty"]
                acc_icon = "✓" if acc >= threshold else "✗"
                click.echo(f"  {label:<25} {diff:<12} {acc_icon} {acc:>7.2f} {comp:>12.2f}")
        
        if ci:
            passed, message = evaluator.check_ci_threshold(results)
            click.echo()
            click.echo(message)
            return 0 if passed else 1
        
        return 0
    
    exit_code = asyncio.run(run())
    sys.exit(exit_code)


# =============================================================================
# Online Evaluation Command
# =============================================================================

@cli.command()
@click.option("--demo", is_flag=True, help="Run demo with sample document")
@click.option("--document", help="Path to document text file to evaluate")
@click.option("--prior-year", help="Path to prior year metrics JSON")
@click.option("--no-telemetry", is_flag=True, help="Disable Application Insights")
@click.option("--confidence-threshold", default=0.7, help="Confidence threshold for SME routing")
def online(
    demo: bool,
    document: str,
    prior_year: str,
    no_telemetry: bool,
    confidence_threshold: float,
):
    """Run online evaluation on a single document."""
    from online_evaluation import OnlineEvaluationMiddleware
    
    async def run():
        # Alert callback
        def on_alert(alert: dict):
            click.echo(f"⚠️  ALERT: {alert['type']}")
            click.echo(f"   Details: {json.dumps(alert, indent=2)}")
        
        # Create middleware
        middleware = OnlineEvaluationMiddleware(
            confidence_threshold=confidence_threshold,
            enable_telemetry=not no_telemetry,
            enable_sme_queue=False,  # Disabled for CLI
            alert_callback=on_alert,
        )
        await middleware.initialize()
        
        # Get document text
        if demo:
            doc_text = """
            Company ABC - Annual Financial Statements 2025
            
            INCOME STATEMENT (in CHF)
            Total Revenue: 15,000,000
            Cost of Goods Sold: 9,000,000
            Gross Profit: 6,000,000
            Operating Expenses: 3,000,000
            Operating Income: 3,000,000
            Interest Expense: 150,000
            Net Income: 2,280,000
            EBITDA: 3,450,000
            
            BALANCE SHEET
            Total Assets: 25,000,000
            Current Assets: 8,000,000
            Total Liabilities: 10,000,000
            Current Liabilities: 3,000,000
            Shareholders' Equity: 15,000,000
            Retained Earnings: 9,500,000
            
            CASH FLOW STATEMENT
            Operating Cash Flow: 3,500,000
            Investing Cash Flow: -2,100,000
            Financing Cash Flow: -800,000
            """
            prior_metrics = {
                "total_revenue": 14_000_000,
                "net_income": 2_100_000,
                "total_assets": 23_000_000,
            }
            doc_id = "demo_001"
        elif document:
            if not Path(document).exists():
                click.echo(f"Error: Document file not found: {document}", err=True)
                return 1
            
            doc_text = Path(document).read_text(encoding="utf-8")
            doc_id = Path(document).stem
            
            prior_metrics = None
            if prior_year:
                prior_metrics = json.loads(Path(prior_year).read_text())
        else:
            click.echo("Error: Either --demo or --document is required", err=True)
            return 1
        
        # Run extraction with evaluation
        result = await middleware.extract_with_evaluation(
            document_text=doc_text,
            document_id=doc_id,
            prior_year_metrics=prior_metrics,
        )
        
        # Print results
        click.echo("\n" + "=" * 50)
        click.echo("EXTRACTION WITH ONLINE EVALUATION")
        click.echo("=" * 50)
        
        click.echo("\nExtracted Metrics:")
        for name, metric in result.extracted_metrics.metrics.items():
            if metric.value:
                click.echo(f"  {name}: {metric.value:,.2f}")
        
        click.echo(f"\nEvaluation Scores:")
        click.echo(f"  Numerical Accuracy: {result.evaluation.numerical_accuracy_score.score:.2f}")
        click.echo(f"  Completeness: {result.evaluation.completeness_score.score:.2f}")
        if result.evaluation.math_consistency_score:
            mc = result.evaluation.math_consistency_score
            click.echo(f"  Math Consistency: {mc.score:.2f}  ({mc.details.get('checks_passed', 0)}/{mc.details.get('checks_total', 0)} checks passed)")
            failed_checks = mc.details.get("failed_checks", [])
            for fc in failed_checks:
                click.echo(f"    ✗ {fc['check_name']}: expected {fc['expected']:,.2f}, got {fc['actual']:,.2f} ({fc['deviation_pct']} off)")
        if result.evaluation.consistency_score:
            click.echo(f"  YoY Consistency: {result.evaluation.consistency_score.score:.2f}")
        click.echo(f"  Overall Confidence: {result.evaluation.overall_confidence:.2f}")
        
        click.echo(f"\nRouting Decision:")
        if result.flagged_for_review:
            click.echo("  ⚠️  FLAGGED FOR SME REVIEW")
            for reason in result.review_reasons:
                click.echo(f"    - {reason}")
        else:
            click.echo("  ✓ Passed - No review needed")
        
        click.echo(f"\nLatency: {result.total_latency_ms:.0f}ms")
        
        await middleware.shutdown()
        return 0
    
    exit_code = asyncio.run(run())
    sys.exit(exit_code)


# =============================================================================
# Compare Command (A/B Testing)
# =============================================================================

@cli.command()
@click.option("--config-a", "config_a", default="baseline", help="Label for configuration A")
@click.option("--config-b", "config_b", default="enhanced", help="Label for configuration B")
@click.option("--dataset", required=True, help="Path to golden dataset")
@click.option("--output", default="results/comparison/", help="Output directory")
@click.option("--no-telemetry", is_flag=True, help="Disable telemetry")
def compare(config_a: str, config_b: str, dataset: str, output: str, no_telemetry: bool):
    """A/B test comparison between two extraction configurations.
    
    Labels the two runs with --config-a and --config-b names.
    In production, each label would map to a different prompt/model config.
    """
    from offline_evaluation import ABTestEvaluator
    from extraction_agent import FinancialMetricsExtractionAgent
    
    click.echo("A/B Test Comparison")
    click.echo("=" * 50)
    click.echo()
    click.echo(f"  Config A: \"{config_a}\"")
    click.echo(f"  Config B: \"{config_b}\"")
    click.echo(f"  Dataset:  {dataset}")
    click.echo()
    click.echo("Note: In this demo both configs use the same agent.")
    click.echo("In production, each config maps to different prompts/models.")
    click.echo()
    
    async def run():
        evaluator = ABTestEvaluator()
        
        # In production, these would be differently configured agents
        agent_a = FinancialMetricsExtractionAgent()
        agent_b = FinancialMetricsExtractionAgent()
        
        await agent_a.initialize()
        await agent_b.initialize()
        
        try:
            results = await evaluator.compare(
                dataset_path=dataset,
                agent_a=agent_a,
                agent_b=agent_b,
                output_path=output,
            )
            
            click.echo("\nComparison Results:")
            click.echo("-" * 40)
            
            for evaluator_name, stats in results["per_evaluator"].items():
                click.echo(f"\n{evaluator_name}:")
                click.echo(f"  {config_a}: {stats['config_a_score']:.2f}")
                click.echo(f"  {config_b}: {stats['config_b_score']:.2f}")
                click.echo(f"  Difference: {stats['difference']:+.2f} ({stats['pct_change']:+.1f}%)")
                winner_label = config_a if stats['winner'] == 'A' else config_b if stats['winner'] == 'B' else 'tie'
                click.echo(f"  Winner: {winner_label}")
            
            click.echo(f"\n{'-' * 40}")
            overall_winner = config_a if results['summary']['winner'] == 'A' else config_b if results['summary']['winner'] == 'B' else 'tie'
            click.echo(f"Overall Winner: {overall_winner}")
            click.echo(f"  {config_a} avg: {results['summary']['config_a_average']:.2f}")
            click.echo(f"  {config_b} avg: {results['summary']['config_b_average']:.2f}")
            click.echo(f"Recommendation: {results['recommendation']}")
            
            click.echo(f"\nResults saved to: {output}")
            
        finally:
            await agent_a.cleanup()
            await agent_b.cleanup()
    
    asyncio.run(run())


# =============================================================================
# Config Command
# =============================================================================

@cli.command()
def config():
    """Display current configuration."""
    cfg = get_config()
    
    click.echo("Current Configuration")
    click.echo("=" * 50)
    
    click.echo("\nAzure AI:")
    click.echo(f"  Project Endpoint: {cfg.azure_ai.project_endpoint or '(not set)'}")
    click.echo(f"  Model Deployment: {cfg.azure_ai.model_deployment_name}")
    click.echo(f"  OpenAI Endpoint: {cfg.azure_ai.openai_endpoint or '(not set)'}")
    
    click.echo("\nThresholds:")
    click.echo(f"  Numerical Accuracy: {cfg.thresholds.numerical_accuracy_threshold}")
    click.echo(f"  Completeness: {cfg.thresholds.completeness_threshold}")
    click.echo(f"  Consistency: {cfg.thresholds.consistency_threshold}")
    click.echo(f"  Confidence: {cfg.thresholds.confidence_threshold}")
    
    click.echo("\nApplication Insights:")
    click.echo(f"  Enabled: {cfg.app_insights.enabled}")
    click.echo(f"  Connection: {'(set)' if cfg.app_insights.connection_string else '(not set)'}")
    
    click.echo("\nSME Queue:")
    click.echo(f"  Enabled: {cfg.sme_queue.enabled}")
    click.echo(f"  Queue Name: {cfg.sme_queue.queue_name}")
    
    # Validate
    errors = cfg.validate()
    if errors:
        click.echo("\n⚠️  Configuration Warnings:")
        for error in errors:
            click.echo(f"  - {error}")
    else:
        click.echo("\n✓ Configuration is valid")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    cli()
