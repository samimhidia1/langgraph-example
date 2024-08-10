from typing import TypedDict, Annotated, Sequence, Literal
from functools import lru_cache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langserve.client import RemoteRunnable

# Define the remote RAG tool
rag_runnable = RemoteRunnable(
    "https://casusragyqouf1pv-casus-mvp-latest.functions.fnc.fr-par.scw.cloud/rag-conversation")


def rag_tool(query: str, chat_history: list) -> str:
    response = rag_runnable.invoke(input={"chat_history": chat_history, "question": query})
    return response


def memo_tool(conversation_history: str) -> str:
    return f"Generated memo based on: {conversation_history}"


def edit_memo(current_memo: str, edit_request: str) -> str:
    return f"Edited memo based on request: {edit_request}\nOriginal memo: {current_memo}"


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
