""" Query the data. """
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import gradio as gr

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.", nargs='?')
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()
    
    if args.ui:
        launch_gradio_ui()
    elif args.query_text:
        query_rag(args.query_text)
    else:
        # Default to launching UI if no arguments provided
        launch_gradio_ui()


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def stream_response(message, history):
    """
    Generator function for streaming responses in Gradio ChatInterface.
    """
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(message, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=message)

    # Create model with streaming enabled
    model = Ollama(model="mistral")
    
    # Stream the response
    response = ""
    for chunk in model.stream(prompt):
        response += chunk
        yield response
    
    # Optionally append sources at the end
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    yield response + f"\n\n**Sources:** {', '.join(sources)}"


def launch_gradio_ui():
    """
    Launch the Gradio ChatInterface.
    """
    chatbot = gr.ChatInterface(
        stream_response,
        textbox=gr.Textbox(
            placeholder="Ask me anything about your documents...",
            container=False,
            autoscroll=True,
            scale=7
        ),
        title="RAG Chatbot",
        description="Ask questions about your documents. The answers are based on the context from your vector database.",
        theme=gr.themes.Soft(),
        examples=[
            "What are the rules for Monopoly?",
            "How do you win in Ticket to Ride?",
            "Tell me about the game setup"
        ],
        cache_examples=False,
    )

    chatbot.launch()


if __name__ == "__main__":
    main()
