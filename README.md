# Simple RAG chatbot with pinecone

This is a basic structure for a Streamlit + FastAPI RAG chatbot project designed to interact with third party data.

## Getting Started

1. Create a `.env` file with the following content:
   ```
    OPENAI_API_KEY=
    PINECONE_API_KEY=
    PINECONE_INDEX_NAME=pooltime-se-chatbot
   ```

2. Create a virtual environment:
   ```bash
   virtualenv venv
   ```

3. Install the required packages:
   ```bash
   (venv) pip install -r requirements.txt
   ```

4. Run the application locally:
   ```bash
   (venv) python chat_cli.py
   ```
