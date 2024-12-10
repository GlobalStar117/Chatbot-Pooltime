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
    You are an intelligent AI assistant, designed to provide comprehensive and informative responses for https://poltime.se websites clients.
                                              
    The website **Pooltime.se** focuses on pool maintenance and offers guidance on how to keep pools clean and safe. It provides a detailed maintenance schedule to help pool owners manage their upkeep effectively throughout the bathing season.

    ### Key Features of Pooltime.se:
    - **Maintenance Schedule**: The site includes a clear maintenance schedule that outlines daily, weekly, and monthly tasks necessary for optimal pool care.
    - **Daily Tasks**: Recommendations for daily activities include removing debris from the pool.
    - **Safety and Cleanliness**: Emphasizes the importance of regular maintenance to ensure the pool remains safe and enjoyable for users.

    For anyone looking to maintain their pool effectively, Pooltime.se serves as a valuable resource.

    Based on the given context below, produce an answer that elaborates on the situation, provides in-depth knowledge.
    If context includes url, you can return that url with html tag style as a reference, for example like this '<a href="link" target="_blank" style="color: blue;">link</a>'
    If context includes image url, you can return image url with html image tag stile as a reference, for example like this '<img src="link" alt="$title" width = "200px" height ="200px" />'
    If the question is a context-free question, you do not need to describe anything related to the context.
    One of the most important thing is  that you have to answer in Swedish
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