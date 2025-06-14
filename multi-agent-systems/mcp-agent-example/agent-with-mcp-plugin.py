# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from dotenv import load_dotenv
from collections import deque

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
# Azure AI imports
from azure.ai.inference.aio import ChatCompletionsClient
from azure.identity import DefaultAzureCredential
from plugins.PublishingPlugin import PublishingPlugin

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
executor_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
utility_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
credential = DefaultAzureCredential()

# Simulate a conversation with the agent
USER_INPUTS = [
    "What are the latest 5 python issues in Microsoft/semantic-kernel?",

]

thread: ChatHistoryAgentThread | None = None

async def main():
    def create_kernel() -> Kernel:
        """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
        kernel = Kernel()
        kernel.add_service(
            service=AzureAIInferenceChatCompletion(
                ai_model_id="utility",
                service_id="utility",
                client=ChatCompletionsClient(
                    endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                    api_version=api_version,
                    credential=credential,
                    credential_scopes=["https://cognitiveservices.azure.com/.default"],
                )
            )
        )
        return kernel

    kernel = create_kernel()

    agent_with_mcp_plugin = ChatCompletionAgent(
        kernel=kernel,
        name="agent-with-mcp-plugin",
        instructions="""
            Your responsibility is to write a scientific poem using plugin available ONLY. Do not generate any content without using the plugin.
            You are able to use the right tools/plugins.
            """,
        plugins=[PublishingPlugin()],
    )

    response = await agent_with_mcp_plugin.get_response(messages="Write a scientific poem about nature.", thread=thread)
    print(f"# {response.name}: {response} ")


if __name__ == "__main__":
    asyncio.run(main())