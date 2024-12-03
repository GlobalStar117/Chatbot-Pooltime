from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from src.vectordb_utils import query_pinecone
import config

def create_rag_chain():
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=config.OPENAI_API_KEY
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Use the following context to answer the question. 
    If you don't know the answer based on the context, just say you don't know.

    Context: {context}

    Question: {question}""")

    # Create the RAG chain
    rag_chain = (
        {"context": lambda x: query_pinecone(x["question"]), "question": lambda x: x["question"]}
        | prompt
        | llm
    )
    
    return rag_chain

def chat_with_rag(question: str):
    chain = create_rag_chain()
    response = chain.invoke({"question": question})
    return response.content