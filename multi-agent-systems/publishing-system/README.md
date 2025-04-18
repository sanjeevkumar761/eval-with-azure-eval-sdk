# MCP client and server with Semantic Kernel plugins  

# How to run it   
1. Start MCP server:  
    ```bash  
    source ../../evalvenv/Scripts/activate  
    cd mcp-server  
    uv run --active mcp-server.py  
    ```
    Wait for the message "Uvicorn running on http://0.0.0.0:8080".  
2. Rename the `.envsample` file to `.env`.  
3. Update the values of the environment variables in the `.env` file (e.g., Azure OpenAI endpoint).  
4. Run the script:  
    ```bash  
    python publishing_system.py  
    ```
5. Input any text on command line to test it.  

# Agent Evaluator for Semantic Kernel

## Overview
This project includes an **AgentEvaluator** to assess the quality of the Semantic Kernel agents' responses. It also demonstrates collaborative interaction between AI agents using the Semantic Kernel framework. The agents, a **Reviewer** and a **Writer**, work together to iteratively improve user-provided content. 

## AgentEvaluator Details

### Initialization
An instance of **AgentEvaluator** is created to assess the quality of the agent's response.

### Evaluation Process
The **AgentEvaluator** is called asynchronously with the following inputs:
- **kernel**: The Semantic Kernel instance managing the agents and their interactions.
- **agent_name**: The name of the agent whose response is being evaluated (e.g., "Reviewer" or "Writer").
- **instructions**: The specific instructions or rules the agent was following while generating the response.
- **input**: The input provided to the agent (e.g., the last message in the conversation).
- **output**: The content generated by the agent in response to the input.

### Logging the Result
The evaluation result is logged for debugging or monitoring purposes, providing insights into how well the agent adhered to its instructions.

## How to Run It
1. Rename the `.envsample` file to `.env`.
2. Update the values of the environment variables in the `.env` file (e.g., Azure OpenAI endpoint).
3. Run the script:
    ```bash
    python agent_collaboration.py
    ```
