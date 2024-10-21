import os


# Initialize Azure OpenAI Connection with your environment variables
model_config = {
    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
}

from azure.ai.evaluation import RelevanceEvaluator, QAEvaluator


# Initialzing Relevance Evaluator
#relevance_eval = RelevanceEvaluator(model_config)
qa_eval = QAEvaluator(model_config)
# Running Relevance Evaluator on single input row
#relevance_score = relevance_eval(
qa_score = qa_eval(
    response="The Alpine Explorer Tent is the most waterproof.",
    context="From the our product list,"
    " the alpine explorer tent is the most waterproof."
    " The Adventure Dining Table has higher weight.",
    query="Which tent is the most waterproof?",
    ground_truth="Alpine Explorer Tent"
)
print(qa_score)

