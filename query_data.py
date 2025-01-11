import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are to act like a lawyer who will assess the situation provided and answer the question based only on the following context:

{context}

---

Provide a clear explanation of how the situation can be resolved, including reasoning and any relevant interpretations of the rules. Answer the question as simply and specifically as possible while ensuring fairness and compliance with the context: {question}
"""

def query_database(query_text):
    """Function to query the database and return a response."""
    try:
        # Get the API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "OpenAI API key is not set in the environment variables.", None, None

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            return "Unable to find matching results.", None, None

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI(openai_api_key=openai_api_key)
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        return response_text, sources, context_text

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

# Streamlit UI
st.title("Query the Database with Streamlit")
query_text = st.text_area("Enter your query:")

if st.button("Submit"):
    with st.spinner("Querying the database..."):
        response, sources, context = query_database(query_text)

    if response:
        st.subheader("Response:")
        st.write(response)

        if context:
            st.subheader("Context:")
            st.write(context)

        if sources:
            st.subheader("Sources:")
            st.write(sources)
    else:
        st.error("No response could be generated.")
