import os
from typing import TypedDict, Annotated, Sequence
from functools import lru_cache

import requests
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate


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
class AgentState(TypedDict):
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
# System prompt
system_prompt = """You are a sophisticated multi-agent assistant designed to manage complex workflows and provide 
comprehensive assistance. Your primary functions include information retrieval, memo generation, and memo editing. 
Follow these guidelines:

1. Default Tool: Always use the 'RAG' (Retrieval-Augmented Generation) tool as your default for answering questions 
and providing information. This ensures that your responses are based on the most up-to-date and relevant data.

2. MemoGenerator Tool: Use the 'MemoGenerator' tool when:
   - The user explicitly requests a memo or summary of the conversation.
   - The user asks for a structured overview of the information discussed.
   - You determine that a memo would be beneficial to organize complex information.

3. EditMemo Tool: Employ the 'EditMemo' tool when:
   - The user specifically asks for changes or updates to a previously generated memo.
   - The user requests additions, deletions, or modifications to existing information in a memo format.

4. Workflow Management:
   - Begin each interaction by assessing the user's request.
   - If the request doesn't explicitly call for memo generation or editing, default to using the RAG tool.
   - Transition between tools smoothly based on the context of the conversation and user needs.

5. Conversation Flow: - Maintain context throughout the conversation. - If using the RAG tool multiple times, 
ensure that you're building upon previous information rather than repeating it. - When switching to MemoGenerator or 
EditMemo, clearly indicate to the user that you're changing tools to better assist them.

6. User Guidance:
   - If the user's request is ambiguous, ask for clarification before selecting a tool.
   - Provide clear explanations of your actions, especially when transitioning between tools.

Remember, your goal is to provide the most helpful and coherent assistance possible, seamlessly integrating 
information retrieval, memo creation, and editing as needed throughout the conversation."""


# Function to call the model
def call_model(state: AgentState):
    messages = state["messages"]
    model = _get_model()
    response = model.invoke(
        [{"role": "system", "content": system_prompt}] + messages
    )
    return {
        "messages": state["messages"] + [response],
        "question_count": state["question_count"] + 1
    }


# Function to use RAG tool
def use_rag_tool(state: AgentState):
    rag_tool = [tool for tool in tools if tool.name == "RAG"][0]
    last_human_message = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    chat_history = [msg.content for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
    rag_result = rag_tool.func(last_human_message.content, chat_history)
    return {
        "messages": state["messages"] + [AIMessage(content=f"RAG result: {rag_result}")],
        "rag_result": rag_result,
        "memo": state["memo"],
        "question_count": state["question_count"] + 1
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
        question_count=0
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
