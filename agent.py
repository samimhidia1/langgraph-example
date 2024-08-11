import json
import os
import time
from typing import TypedDict, Annotated, Sequence, Dict
from functools import lru_cache

import requests
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from typing import List
from dotenv import load_dotenv
from requests import RequestException

load_dotenv()


# Function to get AI response from external API
def get_ai_response(chat_history, question, max_retries=3):
    url = "https://casusragyqouf1pv-casus-mvp-latest.functions.fnc.fr-par.scw.cloud/rag-conversation/invoke"
    headers = {"Content-Type": "application/json"}

    # Format chat_history as a list of tuples
    formatted_chat_history = []
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            formatted_chat_history.append((chat_history[i], chat_history[i + 1]))

    payload = {
        "input": {
            "chat_history": formatted_chat_history,
            "question": question
        },
        "config": {},
        "kwargs": {},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if response.status_code == 422:
                print(f"Server response: {response.text}")
                print(f"Sent payload: {json.dumps(payload, indent=2)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# RAG tool function
def rag_tool(query: str, chat_history: list) -> str:
    try:
        # Ensure chat_history is a list of tuples
        formatted_chat_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                formatted_chat_history.append((chat_history[i], chat_history[i + 1]))

        response = get_ai_response(formatted_chat_history, query)
        return response
    except Exception as e:
        print(f"Error in RAG tool: {e}")
        return "I apologize, but I encountered an error while retrieving information. Let me try to answer based on my general knowledge."

# Function to open and read file content
def open_file(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()


# Function to generate memo
def generate_memo(chat_history):
    model = ChatAnthropic(
        model="claude-3-sonnet-20240320",
        max_tokens=4000,
        temperature=0.2,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    system_prompt = open_file("prompt/claude_system_memo_prompt.txt")
    user_prompt_template = open_file("prompt/claude_user_memo_prompt.txt")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_template)
    ])
    formatted_prompt = prompt.format_messages(CHAT_HISTORY=str(chat_history))
    response = model(formatted_prompt)
    print(response.content)
    return response.content


# Memo tool function
def memo_tool(conversation_history: str) -> str:
    memo = generate_memo(conversation_history)
    print(f"Generated memo: {memo}")
    return memo


# Function to edit memo
def edit_memo(current_memo: str, edit_request: str) -> str:
    model = ChatAnthropic(
        model="claude-3.5-sonnet-20240320",
        max_tokens=8000,
        temperature=0.2,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    system_prompt = open_file("prompt/claude_system_edit_memo_prompt.txt")
    user_prompt_template = open_file("prompt/claude_user_edit_memo_prompt.txt")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_template)
    ])
    formatted_prompt = prompt.format_messages(CURRENT_MEMO=current_memo, EDIT_REQUEST=edit_request)
    response = model(formatted_prompt)
    print(response.content)
    return response.content


# Define tools
tools = [
    Tool.from_function(func=rag_tool, name="RAG", description="Retrieves information to answer questions"),
    Tool.from_function(func=memo_tool, name="MemoGenerator", description="Generates a structured memo"),
    Tool.from_function(func=edit_memo, name="EditMemo", description="Edits the existing memo")
]


# Get model (cached)
@lru_cache(maxsize=1)
def _get_model():
    model = ChatOpenAI(temperature=0, model_name="gpt-4")
    model = model.bind_tools(tools)
    return model


# Define AgentState
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_result: str
    memo: str
    question_count: int


# Function to determine next action
def should_continue(state: AgentState):
    question_count = state.get("question_count", 0)
    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    if question_count < 2:
        return "use_rag"
    elif not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# System prompt
system_prompt = """
You are an advanced AI assistant managing a multi-tool workflow for complex information tasks. Your primary tools are:

1. RAG (Retrieval-Augmented Generation): Your default tool for answering questions and providing up-to-date information.
2. MemoGenerator: For creating structured summaries of conversations or complex information.
3. EditMemo: For modifying existing memos based on user requests.

Guidelines:

1. Tool Selection:
   - Default to RAG for most queries.
   - Use MemoGenerator when explicitly requested or when summarizing complex information would be beneficial.
   - Use EditMemo only when modifying an existing memo.

2. Workflow Management:
   - Assess each user request carefully before selecting a tool.
   - Transition smoothly between tools as needed, explaining your actions to the user.
   - Build upon previous information to avoid repetition.

3. User Interaction:
   - If a request is ambiguous, ask for clarification before proceeding.
   - Clearly communicate your actions, especially when switching tools.
   - Provide concise, relevant responses focused on addressing the user's needs.

4. Information Handling:
   - Ensure all information provided is accurate and up-to-date.
   - Organize complex information logically and coherently.
   - When using RAG multiple times, synthesize information to provide comprehensive answers.

5. Memo Management:
   - Create memos that are clear, structured, and easy to understand.
   - When editing memos, maintain their overall structure and clarity.
   - Confirm user satisfaction after memo creation or editing.

Your goal is to provide efficient, accurate, and helpful assistance while seamlessly integrating the various tools at your disposal. Adapt your communication style to the user's needs and the complexity of the task at hand.
"""

# Function to call the model


def call_model(state: AgentState):
    messages = state["messages"]
    model = _get_model()

    # Convert messages to the format expected by the model
    formatted_messages: List[Dict[str, str]] = [
                                                   {"role": "system", "content": system_prompt}
                                               ] + [
                                                   {
                                                       "role": "user" if isinstance(msg,
                                                                                    HumanMessage) else "assistant" if isinstance(
                                                           msg, AIMessage) else "system",
                                                       "content": msg.content
                                                   }
                                                   for msg in messages
                                               ]

    response = model.invoke(formatted_messages)

    # Convert the response to an AIMessage
    ai_response = AIMessage(content=response.content)

    return {
        "messages": list(state["messages"]) + [ai_response],  # Convert Sequence to list before concatenating
        "question_count": (state.get("question_count") or 0) + 1
    }


# Function to use RAG tool


def use_rag_tool(state):
    rag_tool = [tool for tool in tools if tool.name == "RAG"][0]
    messages = state["messages"]

    # Extract chat history as a list of content strings
    chat_history = [msg.content for msg in messages]

    # Get the last human message
    last_human_message = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))

    try:
        rag_result = rag_tool.func(last_human_message, chat_history)
    except Exception as e:
        print(f"Error in RAG tool: {e}")
        rag_result = "I apologize, but I encountered an error while retrieving information. Let me try to answer based on my general knowledge."

    return {
        "messages": messages + [AIMessage(content=f"RAG result: {rag_result}")],
        "rag_result": rag_result,
        "memo": state.get("memo", ""),
        "question_count": state.get("question_count", 0) + 1
    }


# Create workflow
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))
workflow.add_node("use_rag", use_rag_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "use_rag": "use_rag",
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")
workflow.add_edge("use_rag", "agent")

graph = workflow.compile()


# Function to run the workflow
def run_workflow(user_input: str):
    state = AgentState(
        messages=[HumanMessage(content=user_input)],
        rag_result="",
        memo="",
        question_count=0
    )

    for output in graph.stream(state):
        state.update(output)  # Update the state with new information

        if "messages" in output:
            new_messages = [msg for msg in output["messages"] if msg not in state["messages"]]
            for message in new_messages:
                if isinstance(message, AIMessage):
                    print(f"AI: {message.content}")

            if new_messages and isinstance(new_messages[-1], AIMessage):
                user_input = input("Human: ")
                state["messages"].append(HumanMessage(content=user_input))

        if output.get("memo"):
            print(f"Current Memo: {output['memo']}")

        print(f"Current question count: {state.get('question_count', 'N/A')}")

    print("Workflow completed.")

# Example usage
if __name__ == "__main__":
    print("Welcome to the Multi-Agent Assistant!")
    initial_question = input("Human: ")
    run_workflow(initial_question)
