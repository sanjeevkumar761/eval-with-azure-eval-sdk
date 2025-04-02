
import asyncio
import os
import logging
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

# Azure AI imports
from azure.ai.inference.aio import ChatCompletionsClient
from azure.identity import DefaultAzureCredential

# Custom evaluator import
from evaluators.agent_evaluator import AgentEvaluator

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.WARNING)  # Default logging level
logger = logging.getLogger("agent_collaboration")
logger.setLevel(logging.INFO)  # Application-specific logging level

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
executor_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
utility_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
credential = DefaultAzureCredential()

# Agent names
REVIEWER_NAME = "Reviewer"
WRITER_NAME = "Writer"

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

async def main():
    """Main function to coordinate agent collaboration."""
    # Create a single kernel instance for all agents
    kernel = create_kernel()

    # Define ChatCompletionAgents
    agent_reviewer = ChatCompletionAgent(
        kernel=kernel,
        name=REVIEWER_NAME,
        instructions="""
Your responsibility is to review and identify how to improve user-provided content.
If the user has provided input or direction for content already provided, specify how to address this input.
Never directly perform the correction or provide an example.
Once the content has been updated in a subsequent response, review it again until it is satisfactory.

RULES:
- Only identify suggestions that are specific and actionable. Give only one suggestion.
- Verify previous suggestions have been addressed.
- Never repeat previous suggestions.
""",
    )

    agent_writer = ChatCompletionAgent(
        kernel=kernel,
        name=WRITER_NAME,
        instructions="""
Your sole responsibility is to rewrite content according to review suggestions.
- Always apply all review directions.
- Always revise the content in its entirety without explanation. Produce only 1 sentence.
- Never address the user.
""",
    )

    # Define selection and termination functions
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {REVIEWER_NAME}
- {WRITER_NAME}

Rules:
- If RESPONSE is user input, it is {REVIEWER_NAME}'s turn.
- If RESPONSE is by {REVIEWER_NAME}, it is {WRITER_NAME}'s turn.
- If RESPONSE is by {WRITER_NAME}, it is {REVIEWER_NAME}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    termination_keyword = "yes"
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    # Define history reducer
    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    # Create AgentGroupChat
    chat = AgentGroupChat(
        agents=[agent_reviewer, agent_writer],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=agent_reviewer,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else WRITER_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_reviewer],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=4,
            history_reducer=history_reducer,
        ),
    )

    print(
        "Ready! Type your input, or 'exit' to quit, 'reset' to restart the conversation. "
        "You may pass in a file path using @<path_to_file>."
    )

    is_complete = False

    while not is_complete:
        user_input = input("User > ").strip()
        last_output = user_input
        logger.info(f"Input to Orchestrator: {last_output}")

        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break

        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue

        # Handle file input
        if user_input.startswith("@") and len(user_input) > 1:
            file_name = user_input[1:]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file_name)
            try:
                if not os.path.exists(file_path):
                    print(f"Unable to access file: {file_path}")
                    continue
                with open(file_path, encoding="utf-8") as file:
                    user_input = file.read()
            except Exception:
                print(f"Unable to access file: {file_path}")
                continue

        # Add user input to the chat
        await chat.add_chat_message(message=user_input)

        try:
            async for response in chat.invoke():
                if response is None or not response.name:
                    continue

                logger.info(f"Agent Invoked: {response.name}")
                instructions = (
                    agent_writer.instructions
                    if response.name == WRITER_NAME
                    else agent_reviewer.instructions
                )
                logger.info(f"Agent Instructions: {instructions}")
                logger.info(f"Input to Agent: {last_output}")
                last_output = response.content
                logger.info(f"Output from Agent: {response.content}")

                # Evaluate agent response
                agent_evaluator = AgentEvaluator()
                result = await agent_evaluator(
                    kernel=kernel,
                    agent_name=response.name,
                    instructions=instructions,
                    input=last_output,
                    output=response.content,
                )
                logger.info(f"AgentEvaluator Result: {result}")
        except Exception as e:
            print(f"Error during chat invocation: {e}")

        # Reset the chat's complete flag for the new conversation round
        chat.is_complete = False

if __name__ == "__main__":
    asyncio.run(main())
