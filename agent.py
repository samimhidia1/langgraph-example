from typing import List, TypedDict, Literal, Union
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
import requests
from dotenv import load_dotenv

load_dotenv()

# Définissez d'abord le type Literal pour next_action
NextAction = Literal["rag_search", "draft_memo", "standard_response", "end"]


# Définition de l'état du chat
class ChatState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    chat_history: List[List[str]]
    next_action: NextAction
    legal_issue: str


# Initialisation du modèle Claude 3 Sonnet
model = ChatAnthropic(model='claude-3-5-sonnet-20240620', max_tokens=4000, temperature=0.2)


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


def function_casus_search(state: ChatState):
    question = state["messages"][-1].content

    # Préparer l'historique du chat dans le format attendu par get_ai_response
    chat_history = [(msg.content, state["messages"][i + 1].content)
                    for i, msg in enumerate(state["messages"][:-1:2])]

    try:
        response_json = get_ai_response(chat_history, question)

        # Extraire la réponse du JSON retourné
        ai_response = response_json.get('output', "Désolé, je n'ai pas pu obtenir une réponse.")

        state["messages"].append(AIMessage(content=ai_response))
        state["chat_history"].append([question, ai_response])
    except Exception as e:
        error_message = f"Une erreur s'est produite lors de la recherche : {str(e)}"
        state["messages"].append(AIMessage(content=error_message))
        state["chat_history"].append([question, error_message])

    return state


# Fonction pour générer un mémo
def function_casus_draft(state: ChatState):
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

    response = model(messages)
    state["messages"].append(AIMessage(content=response.content))
    return state


def standard_response(state: ChatState):
    question = state["messages"][-1].content
    response = model([HumanMessage(content=question)])
    state["messages"].append(AIMessage(content=response.content))
    return state


# Fonction pour décider de la prochaine action
def decide_next_action(state: ChatState) -> ChatState:
    last_message = state["messages"][-1].content.lower()

    if any(keyword in last_message for keyword in ["recherche", "information", "question"]):
        state["next_action"] = "rag_search"
    elif any(keyword in last_message for keyword in
             ["mémo", "document", "consultation", "mail", "memon", "mémorandum", "memorandum", "note", "rapport",
              "analyse", "synthèse", "étude", "avis juridique", "conseil", "recommandation"]):
        state["next_action"] = "draft_memo"
    elif "au revoir" in last_message or "terminer" in last_message:
        state["next_action"] = "end"
    else:
        state["next_action"] = "standard_response"

    return state


# Création du graphe
workflow = StateGraph(ChatState)

# Ajout des nœuds
workflow.add_node("decide_action", decide_next_action)
workflow.add_node("rag_search", function_casus_search)
workflow.add_node("draft_memo", function_casus_draft)
workflow.add_node("standard_response", standard_response)

# Configuration des arêtes
workflow.set_entry_point("decide_action")

workflow.add_conditional_edges(
    "decide_action",
    lambda state: state["next_action"],
    {
        "rag_search": "rag_search",
        "draft_memo": "draft_memo",
        "standard_response": "standard_response",
        "end": END
    }
)

for node in ["rag_search", "draft_memo", "standard_response"]:
    workflow.add_edge(node, "decide_action")

# Compilation du graphe
chatbot = workflow.compile()


# Fonction pour exécuter le chat
def chat():
    state: ChatState = {
        "messages": [],
        "chat_history": [],
        "next_action": "standard_response",  # Ceci est maintenant un Literal valide
        "legal_issue": ""
    }
    print("Bot : Bonjour ! Comment puis-je vous aider aujourd'hui ?")
    while True:
        user_input = input("Vous : ")
        state["messages"].append(HumanMessage(content=user_input))

        if any(keyword in user_input.lower() for keyword in
               ["mémo", "document", "consultation", "mail", "memon", "mémorandum", "memorandum", "note", "rapport",
                "analyse", "synthèse", "étude", "avis juridique", "conseil", "recommandation"]):
            state["legal_issue"] = input("Veuillez préciser le problème juridique pour le mémo : ")

        state = chatbot.invoke(state)
        print("Bot :", state["messages"][-1].content)
        if state["next_action"] == "end":
            break


# Lancer le chat
if __name__ == "__main__":
    chat()
