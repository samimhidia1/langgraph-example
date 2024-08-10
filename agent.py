import logging
import os
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


# Function to get AI response from external API
def get_ai_response(chat_history, question):
    url = "https://casusragyqouf1pv-casus-mvp-latest.functions.fnc.fr-par.scw.cloud/rag-conversation/invoke"
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": {
            "chat_history": chat_history,
            "question": question
        },
        "config": {},
        "kwargs": {},
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


# RAG tool function
def rag_tool(query: str, chat_history: list) -> str:
    # Generate a query to send to the RAG API
    response = get_ai_response(chat_history, query)
    return response


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
    question_count = state["question_count"]
    messages = state["messages"]
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


def use_rag_tool(state: AgentState):
    try:
        rag_tool = [tool for tool in tools if tool.name == "RAG"][0]

        # Extract the last human message content
        last_human_message = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
        query = last_human_message.content

        # Extract chat history
        chat_history = [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
            for msg in state["messages"]
            if isinstance(msg, (HumanMessage, AIMessage))
        ]

        # Call the RAG tool with the correct input format
        rag_result = rag_tool.func(query=query, chat_history=chat_history)

        # Convert Sequence to list before concatenating
        updated_messages: List[BaseMessage] = list(state["messages"]) + [AIMessage(content=rag_result)]

        return {
            "messages": updated_messages,
            "rag_result": rag_result,
            "memo": state["memo"],
            "question_count": (state.get("question_count") or 0) + 1
        }
    except Exception as e:
        logging.error(f"Error in use_rag_tool: {str(e)}")
        # Convert Sequence to list before concatenating
        error_messages: List[BaseMessage] = list(state["messages"]) + [AIMessage(
            content="I apologize, but I encountered an error while retrieving information. Could you please rephrase "
                    "your question?")]
        return {
            "messages": error_messages,
            "rag_result": "",
            "memo": state["memo"],
            "question_count": (state.get("question_count") or 0) + 1
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
    initial_state = AgentState(
        messages=[HumanMessage(content=user_input)],
        rag_result="",
        memo="",
        question_count=0  # Ensure this is always initialized
    )

    for output in graph.stream(initial_state):
        if "messages" in output:
            for message in output["messages"]:
                if isinstance(message, AIMessage):
                    print(f"AI: {message.content}")
        if output.get("memo"):
            print(f"Current Memo: {output['memo']}")
        if len(output["messages"]) % 2 == 0:
            user_input = input("Human: ")
            output["messages"].append(HumanMessage(content=user_input))

        # Print the current question count for debugging
        print(f"Current question count: {output['question_count']}")

    print("Workflow completed.")


# Example usage
if __name__ == "__main__":
    print("Welcome to the Multi-Agent Assistant!")
    initial_question = input("Human: ")
    run_workflow(initial_question)
