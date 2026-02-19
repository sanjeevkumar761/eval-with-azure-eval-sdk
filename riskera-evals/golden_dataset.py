"""
Golden Dataset Generator and Loader
====================================

Utilities for creating, managing, and loading golden datasets
for offline evaluation of financial metrics extraction.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

from models import (
    GoldenDataset,
    GoldenDatasetEntry,
    GroundTruthData,
    PriorYearMetrics,
)
from config import REQUIRED_METRICS


# =============================================================================
# Sample Financial Statement Templates
# =============================================================================

SAMPLE_TEMPLATES = [
    # Template 1: Swiss SME format
    """
{company_name} - Jahresabschluss {fiscal_year}

ERFOLGSRECHNUNG (in CHF)
Nettoerlöse aus Lieferungen und Leistungen: {total_revenue:,.0f}
Materialaufwand: {cost_of_goods_sold:,.0f}
Bruttogewinn: {gross_profit:,.0f}
Personalaufwand: {operating_expenses_staff:,.0f}
Übriger Betriebsaufwand: {operating_expenses_other:,.0f}
Betriebsgewinn (EBIT): {operating_income:,.0f}
Finanzaufwand: {interest_expense:,.0f}
Jahresgewinn: {net_income:,.0f}

BILANZ (in CHF)
AKTIVEN
Umlaufvermögen: {current_assets:,.0f}
Anlagevermögen: {fixed_assets:,.0f}
Total Aktiven: {total_assets:,.0f}

PASSIVEN
Kurzfristige Verbindlichkeiten: {current_liabilities:,.0f}
Langfristige Verbindlichkeiten: {long_term_liabilities:,.0f}
Total Fremdkapital: {total_liabilities:,.0f}
Eigenkapital: {shareholders_equity:,.0f}
Total Passiven: {total_passives:,.0f}
""",
    
    # Template 2: German format
    """
{company_name}
Geschäftsbericht {fiscal_year}

Gewinn- und Verlustrechnung
---------------------------
Umsatzerlöse                    EUR {total_revenue:,.2f}
Herstellungskosten             EUR {cost_of_goods_sold:,.2f}
Rohertrag                      EUR {gross_profit:,.2f}
Verwaltungskosten             EUR {operating_expenses:,.2f}
Betriebsergebnis              EUR {operating_income:,.2f}
Zinsen und ähnliche Aufwendungen EUR {interest_expense:,.2f}
Jahresüberschuss              EUR {net_income:,.2f}

Bilanz zum 31.12.{fiscal_year}
-----------------------------
A K T I V A
Umlaufvermögen                EUR {current_assets:,.2f}
Sachanlagen                   EUR {fixed_assets:,.2f}
Bilanzsumme                   EUR {total_assets:,.2f}

P A S S I V A
Kurzfristige Schulden         EUR {current_liabilities:,.2f}
Langfristige Schulden         EUR {long_term_liabilities:,.2f}
Eigenkapital                  EUR {shareholders_equity:,.2f}
Bilanzsumme                   EUR {total_passives:,.2f}
""",

    # Template 3: UK format
    """
{company_name} Ltd.
Annual Accounts for the year ended 31 December {fiscal_year}

PROFIT AND LOSS ACCOUNT
                                        £'000
Turnover                               {total_revenue_k:,.0f}
Cost of sales                         ({cost_of_goods_sold_k:,.0f})
                                       ------
Gross profit                           {gross_profit_k:,.0f}
Administrative expenses               ({operating_expenses_k:,.0f})
                                       ------
Operating profit                       {operating_income_k:,.0f}
Interest payable                      ({interest_expense_k:,.0f})
                                       ------
Profit before taxation                 {profit_before_tax_k:,.0f}
Tax on profit                         ({tax_k:,.0f})
                                       ------
Profit for the financial year          {net_income_k:,.0f}

BALANCE SHEET
                                        £'000
Fixed assets                           {fixed_assets_k:,.0f}
Current assets                         {current_assets_k:,.0f}
                                       ------
Total assets                           {total_assets_k:,.0f}

Current liabilities                    {current_liabilities_k:,.0f}
Long-term liabilities                  {long_term_liabilities_k:,.0f}
                                       ------
Total liabilities                      {total_liabilities_k:,.0f}

Net assets                             {shareholders_equity_k:,.0f}
""",

    # Template 4: Unstructured narrative format (harder)
    """
{company_name} Annual Report {fiscal_year}

Dear Shareholders,

We are pleased to report another successful year. Our company achieved 
total revenues of {total_revenue:,.0f} CHF, representing growth from the 
previous year. After accounting for direct costs of {cost_of_goods_sold:,.0f} CHF,
our gross profit reached {gross_profit:,.0f} CHF.

Operating expenses, including personnel and administrative costs, totaled
{operating_expenses:,.0f} CHF, leaving us with operating income of 
{operating_income:,.0f} CHF. After interest payments of {interest_expense:,.0f} CHF
and taxes, our net profit for the year was {net_income:,.0f} CHF.

Our balance sheet remains strong with total assets of {total_assets:,.0f} CHF.
Current assets amount to {current_assets:,.0f} CHF. On the liability side,
we have current liabilities of {current_liabilities:,.0f} CHF and long-term 
debt of {long_term_liabilities:,.0f} CHF. Shareholders' equity stands at 
{shareholders_equity:,.0f} CHF.

Management Team
{company_name}
""",

    # Template 5: Table-based messy format
    """
{company_name} - FINANCIAL DATA FY{fiscal_year}

                    |  Amount     |  Prior Year  |  Change %
--------------------|-------------|--------------|----------
Revenue             | {total_revenue:>11,.0f} | {prior_revenue:>12,.0f} | {revenue_change:>7.1f}%
COGS               | {cost_of_goods_sold:>11,.0f} | {prior_cogs:>12,.0f} | {cogs_change:>7.1f}%
Gross Margin       | {gross_profit:>11,.0f} | {prior_gross:>12,.0f} | {gross_change:>7.1f}%
OpEx               | {operating_expenses:>11,.0f} | {prior_opex:>12,.0f} | {opex_change:>7.1f}%
EBIT               | {operating_income:>11,.0f} | {prior_ebit:>12,.0f} | {ebit_change:>7.1f}%
Interest           | {interest_expense:>11,.0f} | {prior_interest:>12,.0f} | {interest_change:>7.1f}%
Net Income         | {net_income:>11,.0f} | {prior_net:>12,.0f} | {net_change:>7.1f}%

ASSETS                              LIABILITIES & EQUITY
Total: {total_assets:,.0f}                  Liabilities: {total_liabilities:,.0f}
Current: {current_assets:,.0f}               Current: {current_liabilities:,.0f}
                                    Equity: {shareholders_equity:,.0f}

Key Ratios: Current Ratio = {current_ratio:.2f}, D/E = {debt_to_equity:.2f}
""",
]

COMPANY_NAMES = [
    "Müller GmbH", "Schneider AG", "Fischer & Co", "Weber Holding",
    "Hoffmann Industries", "Klein Manufacturing", "Schmid Trading",
    "Bauer Solutions", "Meyer Technical", "Wagner Systems",
    "Huber Consulting", "Steiner Logistics", "Brunner Foods",
    "Keller Construction", "Graf Textiles", "Baumann Pharma",
]


def _generate_consistent_metrics(
    base_revenue: float,
    year: int,
) -> dict:
    """Generate a consistent set of financial metrics."""
    # Generate with realistic relationships
    total_revenue = base_revenue
    cogs_ratio = random.uniform(0.55, 0.75)
    cost_of_goods_sold = total_revenue * cogs_ratio
    gross_profit = total_revenue - cost_of_goods_sold
    
    opex_ratio = random.uniform(0.15, 0.30)
    operating_expenses = total_revenue * opex_ratio
    operating_income = gross_profit - operating_expenses
    
    interest_ratio = random.uniform(0.01, 0.04)
    interest_expense = total_revenue * interest_ratio
    
    tax_rate = random.uniform(0.15, 0.25)
    profit_before_tax = operating_income - interest_expense
    tax = max(0, profit_before_tax * tax_rate)
    net_income = profit_before_tax - tax
    
    ebitda = operating_income + (total_revenue * random.uniform(0.03, 0.08))
    
    # Balance sheet
    asset_turnover = random.uniform(0.8, 1.5)
    total_assets = total_revenue / asset_turnover
    current_ratio_target = random.uniform(1.2, 2.5)
    current_liabilities = total_revenue * random.uniform(0.15, 0.25)
    current_assets = current_liabilities * current_ratio_target
    
    debt_to_equity_target = random.uniform(0.3, 1.5)
    shareholders_equity = total_assets * (1 / (1 + debt_to_equity_target))
    total_liabilities = total_assets - shareholders_equity
    
    long_term_liabilities = total_liabilities - current_liabilities
    fixed_assets = total_assets - current_assets
    retained_earnings = shareholders_equity * random.uniform(0.5, 0.8)
    
    # Cash flow
    operating_cash_flow = net_income * random.uniform(1.0, 1.5)
    investing_cash_flow = -total_assets * random.uniform(0.05, 0.15)
    financing_cash_flow = -net_income * random.uniform(0.2, 0.5)
    
    # Ratios
    current_ratio = current_assets / current_liabilities if current_liabilities else 0
    debt_to_equity = total_liabilities / shareholders_equity if shareholders_equity else 0
    gross_margin = gross_profit / total_revenue if total_revenue else 0
    net_margin = net_income / total_revenue if total_revenue else 0
    return_on_equity = net_income / shareholders_equity if shareholders_equity else 0
    
    return {
        "total_revenue": total_revenue,
        "cost_of_goods_sold": cost_of_goods_sold,
        "gross_profit": gross_profit,
        "operating_expenses": operating_expenses,
        "operating_income": operating_income,
        "interest_expense": interest_expense,
        "net_income": net_income,
        "ebitda": ebitda,
        "total_assets": total_assets,
        "current_assets": current_assets,
        "fixed_assets": fixed_assets,
        "total_liabilities": total_liabilities,
        "current_liabilities": current_liabilities,
        "long_term_liabilities": long_term_liabilities,
        "shareholders_equity": shareholders_equity,
        "retained_earnings": retained_earnings,
        "operating_cash_flow": operating_cash_flow,
        "investing_cash_flow": investing_cash_flow,
        "financing_cash_flow": financing_cash_flow,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "return_on_equity": return_on_equity,
        "total_passives": total_assets,  # For balance sheet
        "profit_before_tax": profit_before_tax,
        "tax": tax,
    }


def _generate_prior_year_metrics(current: dict, change_factor: float = 0.1) -> dict:
    """Generate prior year metrics with realistic YoY changes."""
    prior = {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and key != "current_ratio" and key != "debt_to_equity":
            # Apply random change
            change = random.uniform(-change_factor, change_factor)
            prior[key] = value / (1 + change) if (1 + change) != 0 else value
        else:
            prior[key] = value
    return prior


def generate_sample_document(
    company_name: str,
    fiscal_year: int,
    template_idx: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Generate a sample financial statement document.
    
    Returns:
        Tuple of (document_text, ground_truth_metrics)
    """
    if metrics is None:
        base_revenue = random.uniform(5_000_000, 100_000_000)
        metrics = _generate_consistent_metrics(base_revenue, fiscal_year)
    
    # Generate prior year for templates that need it
    prior = _generate_prior_year_metrics(metrics)
    
    # Calculate changes for table template
    def calc_change(curr, prev):
        if prev == 0:
            return 0
        return ((curr - prev) / abs(prev)) * 100
    
    template_vars = {
        "company_name": company_name,
        "fiscal_year": fiscal_year,
        **metrics,
        # For UK format (in thousands)
        **{f"{k}_k": v / 1000 for k, v in metrics.items() if isinstance(v, (int, float))},
        # Prior year values
        "prior_revenue": prior["total_revenue"],
        "prior_cogs": prior["cost_of_goods_sold"],
        "prior_gross": prior["gross_profit"],
        "prior_opex": prior["operating_expenses"],
        "prior_ebit": prior["operating_income"],
        "prior_interest": prior["interest_expense"],
        "prior_net": prior["net_income"],
        # Changes
        "revenue_change": calc_change(metrics["total_revenue"], prior["total_revenue"]),
        "cogs_change": calc_change(metrics["cost_of_goods_sold"], prior["cost_of_goods_sold"]),
        "gross_change": calc_change(metrics["gross_profit"], prior["gross_profit"]),
        "opex_change": calc_change(metrics["operating_expenses"], prior["operating_expenses"]),
        "ebit_change": calc_change(metrics["operating_income"], prior["operating_income"]),
        "interest_change": calc_change(metrics["interest_expense"], prior["interest_expense"]),
        "net_change": calc_change(metrics["net_income"], prior["net_income"]),
        "operating_expenses_staff": metrics["operating_expenses"] * 0.7,
        "operating_expenses_other": metrics["operating_expenses"] * 0.3,
        "tax_k": metrics["tax"] / 1000,
        "profit_before_tax_k": metrics["profit_before_tax"] / 1000,
    }
    
    # Select template
    if template_idx is None:
        template_idx = random.randint(0, len(SAMPLE_TEMPLATES) - 1)
    
    template = SAMPLE_TEMPLATES[template_idx]
    document_text = template.format(**template_vars)
    
    # Return only the metrics that are in REQUIRED_METRICS
    ground_truth = {k: v for k, v in metrics.items() if k in REQUIRED_METRICS}
    
    return document_text, ground_truth, prior


def generate_golden_dataset(
    name: str,
    size: int = 50,
    include_prior_year: bool = True,
    output_path: Optional[str] = None,
) -> GoldenDataset:
    """
    Generate a complete golden dataset for evaluation.
    
    Args:
        name: Name for the dataset
        size: Number of entries to generate
        include_prior_year: Whether to include prior year metrics
        output_path: Optional path to save the dataset
    
    Returns:
        GoldenDataset object
    """
    entries = []
    
    for i in range(size):
        company_name = random.choice(COMPANY_NAMES)
        fiscal_year = random.choice([2024, 2025])
        
        # Vary templates to test different formats
        template_idx = i % len(SAMPLE_TEMPLATES)
        
        document_text, ground_truth, prior_metrics = generate_sample_document(
            company_name=company_name,
            fiscal_year=fiscal_year,
            template_idx=template_idx,
        )
        
        document_id = f"doc_{i:04d}_{fiscal_year}"
        
        # Determine difficulty based on template
        difficulty_map = {0: "medium", 1: "medium", 2: "easy", 3: "hard", 4: "hard"}
        difficulty = difficulty_map.get(template_idx, "medium")
        
        entry = GoldenDatasetEntry(
            document_id=document_id,
            raw_text=document_text,
            ground_truth=GroundTruthData(
                document_id=document_id,
                metrics=ground_truth,
                validated_by="synthetic_generator",
                validation_date=datetime.utcnow(),
            ),
            prior_year=PriorYearMetrics(
                document_id=f"{document_id}_prior",
                fiscal_year=fiscal_year - 1,
                metrics={k: v for k, v in prior_metrics.items() if k in REQUIRED_METRICS},
            ) if include_prior_year else None,
            client_industry=random.choice(["Manufacturing", "Services", "Retail", "Technology"]),
            document_format=f"template_{template_idx}",
            difficulty_level=difficulty,
        )
        
        entries.append(entry)
    
    dataset = GoldenDataset(
        name=name,
        description=f"Synthetic golden dataset with {size} entries for evaluation testing",
        entries=entries,
    )
    
    if output_path:
        save_golden_dataset(dataset, output_path)
    
    return dataset


def save_golden_dataset(dataset: GoldenDataset, path: str) -> None:
    """Save golden dataset to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2, default=str)
    
    print(f"Saved golden dataset to {output_path}")


def load_golden_dataset(path: str) -> GoldenDataset:
    """Load golden dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return GoldenDataset(**data)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Golden Dataset Generator")
    parser.add_argument("--generate", action="store_true", help="Generate new dataset")
    parser.add_argument("--size", type=int, default=50, help="Number of entries")
    parser.add_argument("--name", type=str, default="evaluation_dataset_v1", help="Dataset name")
    parser.add_argument("--output", type=str, default="data/golden_dataset.json", help="Output path")
    parser.add_argument("--preview", action="store_true", help="Preview a sample entry")
    
    args = parser.parse_args()
    
    if args.preview:
        doc, metrics, prior = generate_sample_document(
            company_name="Test Company GmbH",
            fiscal_year=2025,
        )
        print("=== Sample Document ===")
        print(doc)
        print("\n=== Ground Truth Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if args.generate:
        dataset = generate_golden_dataset(
            name=args.name,
            size=args.size,
            output_path=args.output,
        )
        print(f"Generated dataset with {dataset.size} entries")
