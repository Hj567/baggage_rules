import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

PROMPT_TEMPLATE = """
You are to act like a lawyer who will assess the situation provided and answer the question based only on the following context:

{context}

---

Provide a clear explanation of how the situation can be resolved, including reasoning and any relevant interpretations of the rules. Answer the question as simply and specifically as possible while ensuring fairness and compliance with the context: {question}
"""

def query_text_file(file_content, query_text):
    """Function to query the text file and return a response."""
    try:
        # Get the API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "OpenAI API key is not set in the environment variables.", None, None

        # Use the file content as the context
        context_text = file_content

        # Prepare the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generate the response
        model = ChatOpenAI(openai_api_key=openai_api_key)
        response_text = model.predict(prompt)

        return response_text, None, context_text

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

# Streamlit UI
st.title("Query the Text File with Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
query_text = st.text_area("Enter your query:")

if uploaded_file is not None:
    # Read file content
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("File Content:", file_content, height=200, disabled=True)

    if st.button("Submit"):
        with st.spinner("Querying the file..."):
            response, _, context = query_text_file(file_content, query_text)

        if response:
            st.subheader("Response:")
            st.write(response)

            if context:
                st.subheader("Context:")
                st.write(context)
        else:
            st.error("No response could be generated.")
else:
    st.info("Please upload a text file to proceed and enter your query.")
