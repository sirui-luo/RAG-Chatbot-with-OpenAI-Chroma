import os
import streamlit as st
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # Add AzureChatOpenAI import
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)

# Load API keys and settings
load_dotenv("api.env")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHROMA_PATH = "chroma/"

# Title
st.title("🤖 RAG Chatbot Demo: Alice in Wonderland")

# Load vector store
embedding = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    deployment=AZURE_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding,
    collection_name="alice_books"
)

# LLM and prompt
llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name="gpt-4o"
)

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {query}
"""
)

# User inputs
query = st.text_input("🔍 Ask a question about the book:")
ground_truth = st.text_area("📘 (Optional) Ground truth answer for evaluation")

# RAG & Evaluation
if query:
    results = db.similarity_search_with_relevance_scores(query, k=5)
    context_docs = [doc.page_content for doc, score in results if score >= 0.4]
    context = "\n\n---\n\n".join(context_docs)

    if context:
        final_prompt = prompt_template.format(context=context, query=query)
        response = llm.invoke(final_prompt).content

        st.subheader("Chatbot Answer Based on Document Context")
        st.write(response)

        st.subheader("Source Context")
        for i, doc in enumerate(context_docs):
            st.code(doc[:500], language="markdown")

        if ground_truth.strip():
            sample_dataset = Dataset.from_list([{
                "question": query,
                "answer": response,
                "contexts": context_docs,
                "ground_truth": ground_truth
            }])

            result = evaluate(
                sample_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ]
            )

            st.subheader("RAGAS Evaluation Output")

            # Convert scores to pretty JSON format
            formatted_result = json.dumps(result.scores, indent=4)

            st.code(formatted_result, language="json")