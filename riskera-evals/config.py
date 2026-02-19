"""
Configuration Module for Financial Metrics Evaluation Framework
================================================================

Centralizes all configuration including Azure AI Foundry settings,
evaluation thresholds, and Application Insights configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AzureAIConfig:
    """Azure AI Foundry and OpenAI configuration."""
    
    project_endpoint: str = field(
        default_factory=lambda: os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
    )
    model_deployment_name: str = field(
        default_factory=lambda: os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    )
    api_version: str = field(
        default_factory=lambda: os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
    )
    
    # For direct Azure OpenAI access (evaluation SDK)
    openai_endpoint: str = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_API_KEY", "")
    )
    openai_deployment: str = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    )


@dataclass
class EvaluationThresholds:
    """Thresholds for evaluation scoring and alerting."""
    
    # Score thresholds (out of 5)
    numerical_accuracy_threshold: float = field(
        default_factory=lambda: float(os.environ.get("NUMERICAL_ACCURACY_THRESHOLD", "3.5"))
    )
    completeness_threshold: float = field(
        default_factory=lambda: float(os.environ.get("COMPLETENESS_THRESHOLD", "4.0"))
    )
    consistency_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONSISTENCY_THRESHOLD", "3.0"))
    )
    groundedness_threshold: float = field(
        default_factory=lambda: float(os.environ.get("GROUNDEDNESS_THRESHOLD", "3.5"))
    )
    
    # Overall confidence threshold for SME routing
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
    )
    
    # Numerical deviation thresholds
    deviation_excellent: float = 0.001  # < 0.1% = Score 5
    deviation_good: float = 0.01       # < 1% = Score 4
    deviation_acceptable: float = 0.05  # < 5% = Score 3
    deviation_marginal: float = 0.10    # < 10% = Score 2
    deviation_poor: float = 0.25        # < 25% = Score 1
    
    # YoY consistency thresholds
    yoy_revenue_max_change: float = 0.50  # 50% max change
    yoy_asset_max_change: float = 0.30    # 30% max change
    yoy_ratio_max_change: float = 0.40    # 40% max change


@dataclass
class ApplicationInsightsConfig:
    """Azure Application Insights configuration for online evaluation."""
    
    connection_string: str = field(
        default_factory=lambda: os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_APP_INSIGHTS", "true").lower() == "true"
    )
    
    # Custom metric names
    metric_prefix: str = "financial_extraction"


@dataclass
class SMEQueueConfig:
    """Azure Service Bus configuration for SME review queue."""
    
    connection_string: str = field(
        default_factory=lambda: os.environ.get("SME_QUEUE_CONNECTION_STRING", "")
    )
    queue_name: str = field(
        default_factory=lambda: os.environ.get("SME_QUEUE_NAME", "sme-review-queue")
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_SME_QUEUE", "false").lower() == "true"
    )


@dataclass
class Config:
    """Main configuration class aggregating all settings."""
    
    azure_ai: AzureAIConfig = field(default_factory=AzureAIConfig)
    thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)
    app_insights: ApplicationInsightsConfig = field(default_factory=ApplicationInsightsConfig)
    sme_queue: SMEQueueConfig = field(default_factory=SMEQueueConfig)
    
    # Data paths
    golden_dataset_path: str = field(
        default_factory=lambda: os.environ.get("GOLDEN_DATASET_PATH", "data/golden_dataset.json")
    )
    results_output_path: str = field(
        default_factory=lambda: os.environ.get("RESULTS_OUTPUT_PATH", "results/")
    )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.azure_ai.project_endpoint and not self.azure_ai.openai_endpoint:
            errors.append("Either AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT must be set")
        
        if self.app_insights.enabled and not self.app_insights.connection_string:
            errors.append("APPLICATIONINSIGHTS_CONNECTION_STRING required when App Insights is enabled")
        
        if self.sme_queue.enabled and not self.sme_queue.connection_string:
            errors.append("SME_QUEUE_CONNECTION_STRING required when SME queue is enabled")
        
        return errors


# Required financial metrics for extraction
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

# Metric categories for reporting
METRIC_CATEGORIES = {
    "income_statement": [
        "total_revenue", "cost_of_goods_sold", "gross_profit",
        "operating_expenses", "operating_income", "interest_expense",
        "net_income", "ebitda"
    ],
    "balance_sheet": [
        "total_assets", "current_assets", "total_liabilities",
        "current_liabilities", "shareholders_equity", "retained_earnings"
    ],
    "cash_flow": [
        "operating_cash_flow", "investing_cash_flow", "financing_cash_flow"
    ],
    "ratios": [
        "current_ratio", "debt_to_equity", "gross_margin",
        "net_margin", "return_on_equity"
    ],
}


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()


def get_model_config():
    """Get Azure OpenAI model configuration for evaluators.
    
    Uses token-based auth (DefaultAzureCredential) by default since
    key-based auth may be disabled on the Azure OpenAI resource.
    """
    from azure.ai.evaluation import AzureOpenAIModelConfiguration
    from azure.identity import DefaultAzureCredential
    
    config = get_config()
    
    endpoint = config.azure_ai.openai_endpoint or config.azure_ai.project_endpoint
    deployment = config.azure_ai.openai_deployment or config.azure_ai.model_deployment_name
    
    # Always use token-based auth — key auth is disabled on this resource
    credential = DefaultAzureCredential(process_timeout=30)
    return AzureOpenAIModelConfiguration(
        azure_endpoint=endpoint,
        credential=credential,
        azure_deployment=deployment,
        api_version=config.azure_ai.api_version,
    )
