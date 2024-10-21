"""
This script evaluates AI models using Azure AI's evaluation tools.
Modules:
    - azure.ai.evaluation: Provides evaluation tools for AI models.
    - answer_length: Custom module for evaluating answer length.
    - json: Standard library for JSON operations.
    - os: Standard library for interacting with the operating system.
    - dotenv: Library for loading environment variables from a .env file.
Environment Variables:
    - SUBSCRIPTION_ID: Azure subscription ID.
    - RESOURCE_GRPUP: Azure resource group name.
    - PROJECT_NAME: Azure project name.
    - AZURE_OPENAI_ENDPOINT: Endpoint for Azure OpenAI.
    - AZURE_OPENAI_API_KEY: API key for Azure OpenAI.
    - AZURE_OPENAI_DEPLOYMENT: Deployment name for Azure OpenAI.
Configuration:
    - azure_ai_project: Dictionary containing Azure project configuration.
    - model_config: Dictionary containing model configuration for Azure OpenAI.
Evaluators:
    - RelevanceEvaluator: Evaluates the relevance of responses.
    - GroundednessEvaluator: Evaluates the groundedness of responses.
Functions:
    - evaluate: Main function to evaluate the AI model using specified evaluators and configuration.
Usage:
    - Ensure all required environment variables are set in a .env file.
    - Modify the `path` variable to point to your dataset in JSONL format.
    - Run the script to perform the evaluation and optionally save the results to a specified output path.
"""
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import RelevanceEvaluator, GroundednessEvaluator
from answer_length import AnswerLengthEvaluator
import json
import os

from dotenv import load_dotenv
load_dotenv()

azure_ai_project = {
    "subscription_id": os.getenv("SUBSCRIPTION_ID"),
    "resource_group_name": os.getenv("RESOURCE_GRPUP"),
    "project_name": os.getenv("PROJECT_NAME"),
} 
# Initialize Azure OpenAI Connection with your environment variables
model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
}

# Initialzing Relevance and Groundedness Evaluators
relevance_eval = RelevanceEvaluator(model_config)
groundedness_eval = GroundednessEvaluator(model_config)


path = "data.jsonl"

result = evaluate(
    azure_ai_project=azure_ai_project,
    data=path,
    evaluators={
        "relevance": relevance_eval,
        "groundedness": groundedness_eval,
#        "answer_length": answer_length
    },
    # column mapping
    evaluator_config={
        "relevance": {
                    "response": "${data.response}",
                    "query": "${data.query}"
                },
        "groundedness_eval": {
                    "response": "${data.response}",
                    "query": "${data.query}",
                    "context": "${data.context}",
                    "ground_truth": "${data.ground_truth}"
                }
    },
    output_path="."

    # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
    #output_path="./myevalresults.json"
)