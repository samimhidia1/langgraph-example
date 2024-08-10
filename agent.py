import os
from typing import TypedDict, Annotated, Sequence
from functools import lru_cache

import requests
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage


def get_ai_response(chat_history, question):
    url = "https://casusragyqouf1pv-casus-mvp-latest.functions.fnc.fr-par.scw.cloud/rag-conversation/invoke"
    headers = {"Content-Type": "application/json"}
    print("chat_history", chat_history)
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


def rag_tool(query: str, chat_history: list) -> str:
    response = get_ai_response(chat_history, query)
    return response


def open_file(file_path):
    """
    opens a txt file and returns the contents
    :param file_path:
    :return:
    """
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()


def generate_memo(chat_history):
    """
    Generate a memo based on the chat history using Anthropic's Claude model.
    """
    # Initialize the ChatAnthropic model
    model = ChatAnthropic(
        model="claude-3-sonnet-20240320",
        max_tokens=4000,
        temperature=0.2,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Load system and user prompts
    system_prompt = open_file("prompt/claude_system_memo_prompt.txt")
    user_prompt_template = open_file("prompt/claude_user_memo_prompt.txt")

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_template)
    ])

    # Format the prompt with the chat history
    formatted_prompt = prompt.format_messages(CHAT_HISTORY=str(chat_history))

    # Generate the response
    response = model(formatted_prompt)

    print(response.content)

    return response.content


def memo_tool(conversation_history: str) -> str:
    memo = generate_memo(conversation_history)
    print(f"Generated memo: {memo}")
    return memo


def edit_memo(current_memo: str, edit_request: str) -> str:
    """
    Edit the current memo based on the user's request.
    """
    # Initialize the ChatAnthropic model
    model = ChatAnthropic(
        model="claude-3.5-sonnet-20240320",
        max_tokens=8000,
        temperature=0.2,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Load system and user prompts
    system_prompt = open_file("prompt/claude_system_edit_memo_prompt.txt")
    user_prompt_template = open_file("prompt/claude_user_edit_memo_prompt.txt")

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_template)
    ])

    # Format the prompt with the current memo and the user's edit request

    formatted_prompt = prompt.format_messages(CURRENT_MEMO=current_memo, EDIT_REQUEST=edit_request)

    # Generate the response

    response = model(formatted_prompt)

    print(response.content)

    return response.content


tools = [
    Tool.from_function(func=rag_tool, name="RAG", description="Retrieves information to answer questions"),
    Tool.from_function(func=memo_tool, name="MemoGenerator", description="Generates a structured memo"),
    Tool.from_function(func=edit_memo, name="EditMemo", description="Edits the existing memo")
]


@lru_cache(maxsize=1)
def _get_model():
    model = ChatOpenAI(temperature=0, model_name="gpt-4")
    model = model.bind_tools(tools)
    return model


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_result: str
    memo: str
    question_count: int


def should_continue(state):
    question_count = state["question_count"]
    messages = state["messages"]
    last_message = messages[-1]

    if question_count < 2:
        return "use_rag"
    elif not last_message.tool_calls:
        return "end"
    else:
        return "continue"


system_prompt = """You are a multi-agent assistant. Manage the workflow and decide which tool to use next: 'RAG', 
'MemoGenerator', 'EditMemo', or finish the conversation. For the first two interactions, always use the RAG tool."""


def call_model(state):
    messages = state["messages"]
    model = _get_model()
    response = model.invoke(
        [{"role": "system", "content": system_prompt}] + messages
    )
    return {"messages": [response]}


def use_rag_tool(state):
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


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
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
        if len(output["messages"]) % 2 == 0:  # Every even number of messages, we ask for user input
            user_input = input("Human: ")
            output["messages"].append(HumanMessage(content=user_input))

    print("Workflow completed.")


# Example usage
if __name__ == "__main__":
    print("Welcome to the Multi-Agent Assistant!")
    initial_question = input("Human: ")
    run_workflow(initial_question)
