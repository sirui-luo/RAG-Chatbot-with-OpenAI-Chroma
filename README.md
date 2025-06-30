# RAG Chatbot Demo with Azure OpenAI & Chroma — Alice in Wonderland

This project demonstrates a small-scale implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using Azure OpenAI, LangChain, and ChromaDB. The chatbot is powered by GPT-4o and responds to user queries based solely on information retrieved from a reference document — *Alice in Wonderland*.

> **Note**: This is a demo project using Alice in Wonderland as a reference. You can easily adapt it to any other corpus (e.g., manuals, documentation, PDFs) by replacing the markdown files in `data/books/`.

<img width="719" alt="image" src="https://github.com/user-attachments/assets/2dc4e3a3-9029-432f-851d-68ed78398b06" />

<img width="730" alt="image" src="https://github.com/user-attachments/assets/07fba8c5-62f0-44c5-8019-7a97f6d10e9a" />

## Core Components

### 1. Document Ingestion & Preprocessing (`build_index.py`)
- Loads `.md` files from the `data/books/` directory using `DirectoryLoader`.
- Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve context.
- Generates dense vector embeddings for each chunk using **Azure OpenAI's `text-embedding-3-large` model**.
- Stores vectors in a persistent **Chroma** vector database under `chroma/`.

---

### 2. RAG Chatbot Interface (`streamlit_app.py`)
- Built with **Streamlit** for interactive querying.
- When a question is entered:
  - Performs **semantic similarity search** via Chroma to retrieve top-5 relevant document chunks (filtered by a relevance threshold ≥ 0.4).
  - Injects those chunks into a custom prompt template.
  - Calls **Azure's GPT-4o** model via `AzureChatOpenAI` to generate a grounded response.

**Displayed in the UI:**
- **Generated Answer**: LLM output grounded in retrieved document context.
- **Source Context**: Document chunks used to generate the answer.
- **Evaluation Output (Optional)**: If a ground truth is provided, the app runs a **RAGAS** evaluation using four core metrics.

---

### 3. Evaluation with RAGAS
- If the user enters a ground truth, the app evaluates the generated answer using:
  - **Faithfulness** – Is the answer grounded in retrieved context?
  - **Answer Relevancy** – Does the answer directly address the question?
  - **Context Precision** – Are the retrieved contexts all relevant?
  - **Context Recall** – Were all needed contexts retrieved?

Results are shown as a clean, structured JSON report in the UI.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-chatbot-alice.git
cd rag-chatbot-alice

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-chatbot-alice.git
cd rag-chatbot-alice

### 2. Create and Fill in Environment Variables
Create a file called api.env in the root directory with the following:
```bash
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
OPENAI_API_KEY=your_openai_key_if_needed_for_ragas

### 3. Install Dependencies
Make sure you're using Python 3.10 or 3.11, then run:
```bash
pip install -r requirements.txt

### 4. Build the Vector Index
```bash
python build_index.py

### 5. Launch the Streamlit App
```bash
streamlit run streamlit_app.py

