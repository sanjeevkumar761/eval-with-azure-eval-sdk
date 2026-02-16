"""
Investment Research Agent with Groundedness Evaluation
Demonstrates that responses are grounded in factual data provided as context.
"""
import os
import json
import time
from datetime import datetime, timezone
from pprint import pprint
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, DatasetVersion
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)
from openai.types.eval_create_params import DataSourceConfigCustom

load_dotenv()

# Environment variables
endpoint = os.environ["PROJECT_ENDPOINT"]
model_deployment = os.environ["MODEL_DEPLOYMENT_NAME"]

# Investment Research Agent name
agent_name = "NewInvestmentResearchAgent1"

print(f"Using PROJECT_ENDPOINT: {endpoint}")
print(f"Using MODEL_DEPLOYMENT_NAME: {model_deployment}")
print(f"Agent: {agent_name}")

# Create project client and OpenAI client
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=endpoint, credential=credential)
client = project_client.get_openai_client()

# Create Investment Research Agent
print("\n=== Creating Investment Research Agent ===")
agent_instructions = """You are an Investment Research Analyst AI assistant. Your role is to:

1. Analyze financial data, market trends, and company fundamentals
2. Provide investment insights based ONLY on the factual data provided
3. Always cite specific numbers, dates, and sources from the context
4. Never make up financial figures or statistics
5. Clearly state when information is not available in the provided context
6. Provide balanced analysis including both opportunities and risks

IMPORTANT: Base your responses ONLY on the factual information provided in the context. 
Do not hallucinate or make up any financial data, statistics, or projections."""

agent = project_client.agents.create_version(
    agent_name=agent_name,
    definition=PromptAgentDefinition(
        model=model_deployment,
        instructions=agent_instructions,
    ),
)
print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")

# Define investment research queries with factual context
# The context contains the "ground truth" facts the agent should use
investment_research_data = [
    {
        "query": "What is Microsoft's financial performance and should I invest?",
        "context": """Microsoft Corporation (MSFT) Q4 FY2025 Financial Report:
- Revenue: $65.6 billion (up 15% YoY)
- Net Income: $24.1 billion (up 18% YoY)  
- Earnings Per Share (EPS): $3.23 (beat estimates of $3.10)
- Azure Cloud Revenue: grew 29% YoY
- Operating Margin: 44.2%
- Cash and Equivalents: $80.5 billion
- Total Debt: $42.3 billion
- Dividend Yield: 0.72%
- P/E Ratio: 35.2
- Market Cap: $3.1 trillion
Analyst Consensus: 42 Buy, 8 Hold, 2 Sell
Price Target Range: $420-$520
Current Price: $445
Risk Factors: AI competition, regulatory scrutiny in EU, enterprise spending slowdown concerns.""",
    },
    {
        "query": "Analyze Tesla's recent performance and investment outlook",
        "context": """Tesla Inc. (TSLA) Q4 FY2025 Financial Report:
- Revenue: $27.2 billion (up 8% YoY)
- Net Income: $2.5 billion (down 12% YoY)
- Vehicle Deliveries: 495,000 units (up 5% YoY)
- Automotive Gross Margin: 17.8% (down from 19.2% YoY)
- Energy Storage Revenue: $2.8 billion (up 45% YoY)
- Free Cash Flow: $1.2 billion
- Cash Position: $26.1 billion
- P/E Ratio: 68.5
- Market Cap: $820 billion
Recent Developments:
- Cybertruck production ramping up to 2,500/week
- FSD v13 released with improved safety metrics
- New Shanghai Megapack factory operational
Risk Factors: Price competition in China, EV demand uncertainty, CEO distraction concerns, regulatory investigations.""",
    },
    {
        "query": "Should I invest in the semiconductor sector, specifically NVIDIA?",
        "context": """NVIDIA Corporation (NVDA) Q4 FY2026 Financial Report:
- Revenue: $38.5 billion (up 94% YoY)
- Data Center Revenue: $34.2 billion (up 115% YoY)
- Net Income: $19.3 billion 
- Gross Margin: 75.2%
- EPS: $4.82 (beat estimates by 12%)
- Cash Position: $31.4 billion
- P/E Ratio: 52.3
- Market Cap: $3.4 trillion
Key Products:
- H200 GPU: 85% market share in AI training
- Blackwell architecture shipping to hyperscalers
- CUDA ecosystem: 4.5 million developers
Industry Context:
- Global AI chip market expected to reach $200B by 2028
- Competition from AMD MI300X and custom chips (Google TPU, Amazon Trainium)
- Taiwan supply chain dependency
- US-China export restrictions impacting China sales by ~$5B annually
Analyst Ratings: 48 Buy, 5 Hold, 1 Sell
Price Target: $950 (current $890)""",
    },
]

# Chat with the agent to collect grounded responses
print("\n=== Generating Investment Research Reports ===")
eval_data = []

for item in investment_research_data:
    query = item["query"]
    context = item["context"]
    
    # Include context in the query for the agent
    full_prompt = f"""Based on the following factual financial data, please provide an investment analysis:

CONTEXT/FACTS:
{context}

QUESTION: {query}

Provide a grounded analysis citing specific figures from the context above."""

    response = client.responses.create(
        extra_body={"agent": {"name": agent_name, "type": "agent_reference"}},
        input=full_prompt,
        store=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"A: {response.output_text[:300]}...")
    
    # Format for groundedness evaluation: query, response, AND context
    eval_data.append({
        "query": query,
        "response": response.output_text,
        "context": context,  # The ground truth facts for groundedness evaluation
    })

# Save evaluation data to JSONL file
eval_file = "investment_eval_data.jsonl"
with open(eval_file, "w") as f:
    for item in eval_data:
        f.write(json.dumps(item) + "\n")
print(f"\n\nSaved {len(eval_data)} research reports to {eval_file}")

# Upload dataset to Foundry
print("\n=== Uploading Dataset to Foundry ===")
dataset: DatasetVersion = project_client.datasets.upload_file(
    name=f"investment-eval-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S_UTC')}",
    version="1",
    file_path=eval_file,
)
print(f"Dataset uploaded (id: {dataset.id})")

# Define data source config with schema INCLUDING context for groundedness
data_source_config = DataSourceConfigCustom(
    {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "response": {"type": "string"},
                "context": {"type": "string"},  # Required for groundedness
            },
            "required": ["query", "response", "context"],
        },
        "include_sample_schema": True,
    }
)

# Define testing criteria with GROUNDEDNESS evaluator
testing_criteria = [
    # GROUNDEDNESS - Key evaluator to verify response is grounded in context
    {
        "type": "azure_ai_evaluator",
        "name": "groundedness",
        "evaluator_name": "builtin.groundedness",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "context": "{{item.context}}",  # The factual data to ground against
        },
        "initialization_parameters": {"deployment_name": model_deployment},
    },
    # COHERENCE - Logical flow of response
    {
        "type": "azure_ai_evaluator",
        "name": "coherence",
        "evaluator_name": "builtin.coherence",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        },
        "initialization_parameters": {"deployment_name": model_deployment},
    },
    # FLUENCY - Language quality
    {
        "type": "azure_ai_evaluator",
        "name": "fluency",
        "evaluator_name": "builtin.fluency",
        "data_mapping": {
            "response": "{{item.response}}",
        },
        "initialization_parameters": {"deployment_name": model_deployment},
    },
    # RELEVANCE - Response addresses the query
    {
        "type": "azure_ai_evaluator",
        "name": "relevance",
        "evaluator_name": "builtin.relevance",
        "data_mapping": {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
            "context": "{{item.context}}",
        },
        "initialization_parameters": {"deployment_name": model_deployment},
    },
]

# Create evaluation
print("\n=== Creating Evaluation with Groundedness ===")
eval_object = client.evals.create(
    name=f"Investment Research Groundedness - {agent_name}",
    data_source_config=data_source_config,
    testing_criteria=testing_criteria,  # type: ignore
)
print(f"Evaluation created (id: {eval_object.id}, name: {eval_object.name})")

# Create evaluation run with dataset ID
print("\n=== Creating Evaluation Run ===")
eval_run = client.evals.runs.create(
    eval_id=eval_object.id,
    name=f"groundedness-run-{datetime.now(timezone.utc).strftime('%H%M%S')}",
    metadata={"agent": agent_name, "scenario": "investment-research-groundedness"},
    data_source=CreateEvalJSONLRunDataSourceParam(
        type="jsonl",
        source=SourceFileID(type="file_id", id=dataset.id if dataset.id else ""),
    ),
)
print(f"Evaluation run created (id: {eval_run.id})")

# Wait for completion
print("\n=== Waiting for Evaluation to Complete ===")
while True:
    run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
    print(f"Status: {run.status}")
    if run.status == "completed" or run.status == "failed":
        break
    time.sleep(5)

# Print results
print("\n" + "="*70)
print("INVESTMENT RESEARCH AGENT - GROUNDEDNESS EVALUATION RESULTS")
print("="*70)

if run.status == "completed":
    output_items = list(client.evals.runs.output_items.list(run_id=run.id, eval_id=eval_object.id))
    print(f"\nEvaluated {len(output_items)} investment research reports\n")
    
    # Extract and display scores
    for i, item in enumerate(output_items, 1):
        print(f"Report {i}: {item.datasource_item.get('query', 'N/A')[:50]}...")
        for result in item.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            score = f"{result.score:.1f}" if result.score is not None else "N/A"
            print(f"  {result.name:15} Score: {score:5} {status}")
        print()
    
    print("="*70)
    print("GROUNDEDNESS EXPLANATION:")
    print("- Score 1-2: Response contains claims NOT supported by the context")
    print("- Score 3: Response is partially grounded, some claims unsupported")
    print("- Score 4-5: Response is fully grounded in the provided factual context")
    print("="*70)
    
    if run.report_url:
        print(f"\n✓ View detailed results in Foundry portal:")
        print(f"  {run.report_url}")
else:
    print(f"Evaluation failed: {run}")

print("\n" + "="*70)
print(f"Agent: {agent_name}")
print(f"Evaluation ID: {eval_object.id}")
print(f"Run ID: {run.id}")
print("="*70)
