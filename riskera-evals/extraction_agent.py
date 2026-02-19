"""
Financial Metrics Extraction Agent
===================================

Azure AI Foundry agent for extracting financial metrics from
non-standardized financial statements.

Uses Microsoft Agent Framework with custom tools for:
- Metric extraction from raw document text
- Mathematical validation and correction
- Prior year comparison
"""

import os
import json
import asyncio
from typing import Optional, Any
from datetime import datetime

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    Agent,
    AgentThread,
    ThreadMessage,
    MessageRole,
    ToolSet,
    FunctionTool,
)

from config import get_config, REQUIRED_METRICS
from models import FinancialMetric, FinancialMetricsSet, Currency


# =============================================================================
# Extraction Prompt Templates
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a financial analyst expert specializing in extracting structured financial metrics from corporate financial statements.

Your task is to extract specific financial metrics from the provided document text. These documents are from small to medium-sized corporate clients and may not follow standard formats like US GAAP or IFRS.

CRITICAL INSTRUCTIONS:
1. Extract values that are explicitly stated in the document
2. For standard financial ratios (current_ratio, debt_to_equity, gross_margin, net_margin, return_on_equity), CALCULATE them from the extracted values if not explicitly stated
3. For EBITDA, calculate as operating_income + depreciation_and_amortization if not stated. If depreciation is unknown, estimate EBITDA = operating_income * 1.15
4. If a metric cannot be found or calculated, mark it as null
5. Pay attention to units (thousands, millions, billions) and currency
6. When numbers appear multiple times, prefer values from summary tables or clearly labeled sections

OUTPUT FORMAT:
Return a JSON object with the following structure for each metric:
{
    "metric_name": {
        "value": <number or null>,
        "currency": "<USD|EUR|GBP|CHF|OTHER>",
        "unit": "<thousands|millions|billions|null>",
        "raw_text": "<exact text from document>",
        "confidence": <0.0-1.0>,
        "source_location": "<description of where found>"
    }
}

REQUIRED METRICS TO EXTRACT:
""" + "\n".join(f"- {m}" for m in REQUIRED_METRICS)


VALIDATION_PROMPT = """Review the extracted financial metrics for mathematical consistency.

Check the following relationships:
1. gross_profit = total_revenue - cost_of_goods_sold
2. operating_income = gross_profit - operating_expenses
3. net_income should be less than operating_income (after interest/taxes)
4. total_assets = total_liabilities + shareholders_equity
5. current_ratio = current_assets / current_liabilities
6. gross_margin = gross_profit / total_revenue
7. net_margin = net_income / total_revenue

For any inconsistencies:
1. Flag the discrepancy
2. Suggest which value might be incorrect
3. Do NOT modify values, only flag issues

Output format:
{
    "validation_passed": true/false,
    "issues": [
        {
            "check": "description of check",
            "expected": value,
            "actual": value,
            "severity": "low|medium|high",
            "recommendation": "suggested action"
        }
    ]
}
"""


# =============================================================================
# Extraction Agent Class
# =============================================================================

class FinancialMetricsExtractionAgent:
    """
    Agent for extracting financial metrics using Azure AI Foundry.
    
    Supports both persistent (cloud) and ephemeral (local) agent modes.
    """
    
    def __init__(
        self,
        use_persistent_agent: bool = False,
        agent_id: Optional[str] = None,
    ):
        self.config = get_config()
        self.use_persistent_agent = use_persistent_agent
        self.agent_id = agent_id
        
        # Initialize Azure AI client with extended timeout for Azure CLI
        self.credential = DefaultAzureCredential(
            process_timeout=30,
        )
        
        if self.config.azure_ai.project_endpoint:
            self.project_client = AIProjectClient(
                credential=self.credential,
                endpoint=self.config.azure_ai.project_endpoint,
            )
        else:
            self.project_client = None
        
        self._agent: Optional[Agent] = None
    
    async def initialize(self) -> None:
        """Initialize the extraction agent."""
        if self.use_persistent_agent and self.agent_id and self.project_client:
            try:
                self._agent = self.project_client.agents.get_agent(self.agent_id)
            except Exception as e:
                print(f"Warning: Could not load agent, falling back to direct OpenAI: {e}")
                self.project_client = None
        elif self.project_client:
            try:
                self._agent = self._create_agent()
            except Exception as e:
                print(f"Warning: Could not create agent, falling back to direct OpenAI: {e}")
                self.project_client = None
    
    def _create_agent(self) -> Agent:
        """Create new extraction agent."""
        agent = self.project_client.agents.create_agent(
            model=self.config.azure_ai.model_deployment_name,
            name="Financial Metrics Extractor",
            instructions=EXTRACTION_SYSTEM_PROMPT,
        )
        return agent
    
    def _get_tools(self) -> list:
        """Get tool definitions for the agent."""
        return [
            FunctionTool(
                name="validate_metrics",
                description="Validate extracted metrics for mathematical consistency",
                parameters={
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "object",
                            "description": "Dictionary of extracted metrics"
                        }
                    },
                    "required": ["metrics"]
                }
            ),
            FunctionTool(
                name="compare_with_prior_year",
                description="Compare current metrics with prior year for anomaly detection",
                parameters={
                    "type": "object",
                    "properties": {
                        "current_metrics": {
                            "type": "object",
                            "description": "Current year metrics"
                        },
                        "prior_metrics": {
                            "type": "object",
                            "description": "Prior year metrics"
                        }
                    },
                    "required": ["current_metrics", "prior_metrics"]
                }
            ),
        ]
    
    async def extract_metrics(
        self,
        document_text: str,
        document_id: str,
        fiscal_year: int,
        prior_year_metrics: Optional[dict[str, float]] = None,
    ) -> FinancialMetricsSet:
        """
        Extract financial metrics from document text.
        
        Args:
            document_text: Raw text extracted from financial statement
            document_id: Unique identifier for the document
            fiscal_year: Fiscal year of the statement
            prior_year_metrics: Optional prior year values for context
        
        Returns:
            FinancialMetricsSet with extracted metrics
        """
        # Build extraction prompt
        prompt = self._build_extraction_prompt(document_text, prior_year_metrics)
        
        # Call the model
        extracted_data = await self._call_model(prompt)
        
        # Parse response into structured format
        metrics_set = self._parse_extraction_response(
            response=extracted_data,
            document_id=document_id,
            fiscal_year=fiscal_year,
        )
        
        return metrics_set
    
    def _build_extraction_prompt(
        self,
        document_text: str,
        prior_year_metrics: Optional[dict[str, float]] = None,
    ) -> str:
        """Build the extraction prompt with document and context."""
        prompt_parts = [
            "Extract financial metrics from the following document:",
            "",
            "=== DOCUMENT START ===",
            document_text[:50000],  # Limit document size
            "=== DOCUMENT END ===",
            "",
        ]
        
        if prior_year_metrics:
            prompt_parts.extend([
                "Prior year metrics for reference (use these to validate reasonableness):",
                json.dumps(prior_year_metrics, indent=2),
                "",
            ])
        
        prompt_parts.append(
            "Extract all available metrics and return as JSON. "
            "Mark any metric not found in the document as null."
        )
        
        return "\n".join(prompt_parts)
    
    async def _call_model(self, prompt: str) -> dict[str, Any]:
        """Call the model and get extraction response."""
        # Prefer direct OpenAI call (faster, works with token auth)
        if self.config.azure_ai.openai_endpoint or self.config.azure_ai.project_endpoint:
            return await self._call_openai_direct(prompt)
        elif self.project_client and self._agent:
            # Use agent via Azure AI Projects SDK
            from azure.ai.agents.models import AgentThreadCreationOptions, ThreadMessageOptions
            run = self.project_client.agents.create_thread_and_process_run(
                agent_id=self._agent.id,
                thread=AgentThreadCreationOptions(
                    messages=[ThreadMessageOptions(role="user", content=prompt)]
                ),
            )
            # Extract response from the last message
            if hasattr(run, 'last_message') and run.last_message:
                return self._parse_json_from_response(run.last_message.content)
            return {}
        else:
            raise RuntimeError("No Azure OpenAI endpoint or AI Projects client configured")
    
    async def _call_openai_direct(self, prompt: str) -> dict[str, Any]:
        """Direct Azure OpenAI call without agent framework."""
        from openai import AsyncAzureOpenAI
        from azure.identity import get_bearer_token_provider
        
        endpoint = self.config.azure_ai.openai_endpoint or self.config.azure_ai.project_endpoint
        
        # Prefer token-based auth (works even when key auth is disabled on the resource)
        try:
            token_provider = get_bearer_token_provider(
                self.credential, "https://cognitiveservices.azure.com/.default"
            )
            client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=self.config.azure_ai.api_version,
            )
        except Exception:
            client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=self.config.azure_ai.openai_api_key,
                api_version=self.config.azure_ai.api_version,
            )
        
        response = await client.chat.completions.create(
            model=self.config.azure_ai.openai_deployment or self.config.azure_ai.model_deployment_name,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent extraction
        )
        
        content = response.choices[0].message.content
        return self._parse_json_from_response(content)
    
    def _parse_json_from_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from model response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {}
    
    def _parse_extraction_response(
        self,
        response: dict[str, Any],
        document_id: str,
        fiscal_year: int,
    ) -> FinancialMetricsSet:
        """Parse model response into FinancialMetricsSet."""
        metrics = {}
        
        for metric_name in REQUIRED_METRICS:
            metric_data = response.get(metric_name, {})
            
            if isinstance(metric_data, dict):
                value = metric_data.get("value")
                currency_str = metric_data.get("currency", "OTHER")
                
                # Parse currency
                try:
                    currency = Currency(currency_str) if currency_str else None
                except ValueError:
                    currency = Currency.OTHER
                
                metrics[metric_name] = FinancialMetric(
                    name=metric_name,
                    value=float(value) if value is not None else None,
                    currency=currency,
                    unit=metric_data.get("unit"),
                    raw_text=metric_data.get("raw_text"),
                    confidence=float(metric_data.get("confidence", 1.0)),
                    source_location=metric_data.get("source_location"),
                )
            elif isinstance(metric_data, (int, float)):
                # Simple numeric value
                metrics[metric_name] = FinancialMetric(
                    name=metric_name,
                    value=float(metric_data),
                )
        
        return FinancialMetricsSet(
            document_id=document_id,
            fiscal_year=fiscal_year,
            metrics=metrics,
            extraction_timestamp=datetime.utcnow(),
        )
    
    async def validate_extraction(
        self,
        metrics_set: FinancialMetricsSet,
    ) -> dict[str, Any]:
        """
        Validate extracted metrics for mathematical consistency.
        
        Args:
            metrics_set: Extracted metrics to validate
        
        Returns:
            Validation results with any flagged issues
        """
        metrics_dict = {
            name: metric.normalized_value
            for name, metric in metrics_set.metrics.items()
            if metric.value is not None
        }
        
        prompt = f"""
{VALIDATION_PROMPT}

Metrics to validate:
{json.dumps(metrics_dict, indent=2)}
"""
        
        result = await self._call_model(prompt)
        return result
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self._agent and self.project_client and not self.use_persistent_agent:
            try:
                self.project_client.agents.delete_agent(self._agent.id)
            except Exception:
                pass  # Ignore cleanup errors


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_financial_metrics(
    document_text: str,
    document_id: str,
    fiscal_year: int,
    prior_year_metrics: Optional[dict[str, float]] = None,
) -> FinancialMetricsSet:
    """
    Convenience function for one-off metric extraction.
    
    Args:
        document_text: Raw text from financial statement
        document_id: Unique document identifier
        fiscal_year: Fiscal year of the statement
        prior_year_metrics: Optional prior year values
    
    Returns:
        Extracted metrics as FinancialMetricsSet
    """
    agent = FinancialMetricsExtractionAgent()
    await agent.initialize()
    
    try:
        result = await agent.extract_metrics(
            document_text=document_text,
            document_id=document_id,
            fiscal_year=fiscal_year,
            prior_year_metrics=prior_year_metrics,
        )
        return result
    finally:
        await agent.cleanup()


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Simple test
    test_doc = """
    COMPANY XYZ - ANNUAL FINANCIAL STATEMENTS FY2025
    
    INCOME STATEMENT
    Total Revenue: CHF 15,500,000
    Cost of Goods Sold: CHF 9,300,000
    Gross Profit: CHF 6,200,000
    Operating Expenses: CHF 3,100,000
    Operating Income: CHF 3,100,000
    Interest Expense: CHF 200,000
    Net Income: CHF 2,320,000
    
    BALANCE SHEET
    Total Assets: CHF 25,000,000
    Current Assets: CHF 8,000,000
    Total Liabilities: CHF 12,000,000
    Current Liabilities: CHF 4,000,000
    Shareholders' Equity: CHF 13,000,000
    """
    
    async def test():
        result = await extract_financial_metrics(
            document_text=test_doc,
            document_id="test_001",
            fiscal_year=2025,
        )
        print("Extracted metrics:")
        for name, metric in result.metrics.items():
            if metric.value:
                print(f"  {name}: {metric.value} {metric.currency}")
    
    asyncio.run(test())
