from typing import TypedDict, Literal, List, Union
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
import requests


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define the state
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    chat_history: List[List[str]]
    next_action: Literal["rag_search", "draft_memo", "standard_response", "end"]
    legal_issue: str
    config: GraphConfig


# Initialize the model
model = ChatAnthropic(model='claude-3-5-sonnet-20240620')


# Define utility functions
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


# Define node functions
def call_model(state: AgentState) -> AgentState:
    if not state["messages"]:
        # Gérer le cas où il n'y a pas de messages
        return state

    last_message = state["messages"][-1]
    last_message_content = last_message.get('content', '') if isinstance(last_message, dict) else last_message.content

    if state["next_action"] == "draft_memo":
        system_prompt = """
        You are a highly experienced legal expert and tax advisor with extensive knowledge of French tax law and corporate regulations. You have been asked to prepare a comprehensive legal memorandum regarding <legal_issue>.
    
        In formulating your analysis and recommendations, you must base all of your reasoning and factual assertions on the information provided in the <chat_history>. Please ensure that your memo is grounded in and consistent with the details discussed in our previous conversation.
    
        Please structure your memo as follows:
    
        ## 1. Introduction
    
        - Briefly describe the context and purpose of the memo
        - Clearly state the legal issue(s) to be addressed
        - Provide an overview of the memo's structure
    
        ## 2. Analyse
    
        ### 2.1. Principe général
    
        - Explain the fundamental principles related to <legal_issue>
        - Reference relevant articles from the Code Général des Impôts (CGI) or other applicable laws, regulations, and jurisprudence
    
        ### 2.2. Conditions et règles spécifiques
    
        - Detail the specific conditions and rules applicable to <legal_issue>
        - Use numbered or bulleted lists for clarity when appropriate
        - Provide examples to illustrate complex points
    
        ### 2.3. Cas particuliers
    
        - Identify and explain any special cases or exceptions related to <legal_issue>
        - Discuss how these cases might affect the general rules
    
        ### 2.4. Application pratique
    
        - Apply the legal analysis to the specific circumstances of <client_name>
        - Address each of the following questions:
          <specific_questions>
    
        ## 3. Recommandations
    
        - Provide clear, actionable recommendations based on your analysis
        - Consider both short-term and long-term implications
        - Highlight potential strategies for optimization within the legal framework
    
        ## 4. Conclusion
    
        - Summarize the key points of your analysis
        - Restate the most important recommendations
        - Highlight any areas of uncertainty or potential risks
    
        ## 5. Références
    
        - List relevant legal texts, articles, and official documents cited in the memo with proper formatting and provide clickable and complete URLs
        - Include links to online resources and full clickable URLs
        - Format references consistently, providing full citations
    
        Throughout your memo, ensure that you:
    
        - Use formal, professional language appropriate for legal communication in French
        - Provide detailed explanations and reasoning for your conclusions
        - Cite relevant laws, regulations, and administrative guidelines where necessary
        - Use numbered sections and subsections with a well-structured comprehensive and professional answer for clear organization
        - Bold or italicize key points for emphasis
        - Use tables or charts if they can help clarify complex information
        - Use line breaks for better readability and organization
        - Use Markdown formatting for headings, lists, and emphasis
    
        Before finalizing the memo:
        - Double-check all legal references for accuracy
        - Ensure consistency in terminology and formatting throughout the document
        - Verify that all <specific_questions> have been adequately addressed
        """
        user_message = f"""
        Generate an outstanding memo using the entire context provided below, including the chat history and legal issue. Be comprehensive, detailed, and provide rich and accurate answers.
    
        <chat_history>
        {state['chat_history']}
        </chat_history>
    
        <legal_issue>
        {state['legal_issue']}
        </legal_issue>
    
        This is the specific legal issue you need to address with verbosity, precision, and contextual accuracy. Use the information from the chat history to inform your analysis and recommendations.
    
        Begin your response immediately after these instructions, starting with the document title.
    
        Ensure your answer is comprehensive, well-structured, and rich in accurate legal references, including all relevant URLs from the provided context and chat history.
    
        Use formal, professional language and provide detailed explanations and reasoning for your conclusions as the best top-notch tax lawyer in the world would do.
    
        Walk me through this context in manageable parts step by step, summarizing and analyzing as we go. Make sure to integrate relevant information from the chat history into your analysis.
    
        Begin your response immediately after these instructions, starting with the document title.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        response = model.invoke(messages)
    else:
        response = model.invoke([HumanMessage(content=last_message_content)])

    response_content = response.content if hasattr(response, 'content') else str(response)
    state["messages"].append(AIMessage(content=response_content))
    return state


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    if not state["messages"]:
        return "continue"

    last_message = state["messages"][-1]
    last_message_content = last_message.content.lower()

    if any(keyword in last_message_content for keyword in ["recherche", "information", "question"]):
        state["next_action"] = "rag_search"
        return "continue"
    elif any(keyword in last_message_content for keyword in ["mémo", "document", "consultation", "mail", "memon", "mémorandum", "memorandum", "note", "rapport", "analyse", "synthèse", "étude", "avis juridique", "conseil", "recommandation"]):
        state["next_action"] = "draft_memo"
        return "continue"
    elif "au revoir" in last_message_content or "terminer" in last_message_content:
        return "end"
    else:
        state["next_action"] = "standard_response"
        return "continue"

def tool_node(state: AgentState):
    if state["next_action"] == "rag_search":
        question = state["messages"][-1].content
        chat_history = [(msg.content, state["messages"][i + 1].content)
                        for i, msg in enumerate(state["messages"][:-1:2])]
        try:
            response_json = get_ai_response(chat_history, question)
            ai_response = response_json.get('output', "Désolé, je n'ai pas pu obtenir une réponse.")
            state["messages"].append(AIMessage(content=ai_response))
            state["chat_history"].append([question, ai_response])
        except Exception as e:
            error_message = f"Une erreur s'est produite lors de la recherche : {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            state["chat_history"].append([question, error_message])
    return state




# Add configuration to the initial state when you create the graph
initial_state = {
    "messages": [],
    "chat_history": [],
    "next_action": "standard_response",
    "legal_issue": "",
    "config": {"model_name": "anthropic"}  # ou "openai" selon votre choix
}

# Define the graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Add normal edge
workflow.add_edge("action", "agent")

# Compile the graph
graph = workflow.compile()
