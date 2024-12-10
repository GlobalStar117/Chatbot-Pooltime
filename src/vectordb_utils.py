import json
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import csv
import uuid
from dotenv import load_dotenv
import config
import time
from llama_index.core import SimpleDirectoryReader
from pathlib import Path



pc = Pinecone(api_key=config.PINECONE_API_KEY)
dims = 3072
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  # us-east-1
)

# check if index already exists (it shouldn't if this is first time)
existing_indexes = pc.list_indexes()
# if config.PINECONE_INDEX_NAME not in existing_indexes:
#     # if does not exist, create index
#     pc.create_index(
#         name=config.PINECONE_INDEX_NAME,
#         dimension=dims,  # dimensionality of embed 3
#         metric='cosine',
#         spec=spec
#     )
#     # wait for index to be initialized
#     while not pc.describe_index(config.PINECONE_INDEX_NAME).status['ready']:
#         time.sleep(1)

# connect to index
index = pc.Index(config.PINECONE_INDEX_NAME)

embed_model = OpenAIEmbeddings(
    model=config.ModelType.embedding,
    openai_api_key=config.OPENAI_API_KEY
)

# embed and index all our our data!
def import_json_to_vector(json_file_path, data_type="website", metadata_processor=None):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        reader = json.load(file)
        
        for ind, item in enumerate(reader):
            print(f"Processing item {ind}")
            embedding_id = str(uuid.uuid4())
            
            # Prepare the text to be embedded based on data type
            if data_type == "website":
                text_to_embed = f"Title: {item['title']}\n{item['content']}"
                metadata = {
                    'title': item['title'],
                    'content' : item['content'],
                    'link': item['link'],
                    'featured_image': item['featured_image'],
                    'type': 'website_content'
                }
            else:  # product data
                text_to_embed = f"Name: {item['title']}\n{item['content']}"
                metadata = {
                    'name': item['title'],
                    'description' : item['content'],
                    'price': item['price'],
                    'regular_price': item['regular_price'],
                    'sale_price':item['sale_price'],
                    'link': item['link'],
                    'image': item['featured_image'],
                    'gallery_images' : item['gallery_images'],
                    'type': 'product'
                }
            
            if metadata_processor:
                metadata = metadata_processor(metadata)
            
            vector = [{
                'id': embedding_id,
                'values': embed_model.embed_documents([text_to_embed])[0],
                'metadata': metadata
            }]
            
            index.upsert(vectors=vector)
            print("Item processed successfully")

    print(f"JSON data imported successfully into pinecone vector database. Data type: {data_type}")

def import_pdfs_to_vector(pdf_directory: str):
    """
    Import PDF files from a directory into Pinecone vector database
    """
    # Load PDFs from directory
    documents = SimpleDirectoryReader(
        input_dir=pdf_directory,
        filename_as_id=True
    ).load_data()
    
    print(f"Found {len(documents)} PDF documents")
    
    for doc in documents:
        print(f"Processing document: {doc.doc_id}")
        embedding_id = str(uuid.uuid4())
        
        # Create chunks of ~1000 characters from the document
        text_chunks = [doc.text[i:i+1000] for i in range(0, len(doc.text), 1000)]
        
        for chunk_idx, chunk in enumerate(text_chunks):
            vector = [{
                'id': f"{embedding_id}_{chunk_idx}",
                'values': embed_model.embed_documents([chunk])[0],
                'metadata': {
                    'file_name': doc.doc_id,
                    'chunk_index': chunk_idx,
                    'type': 'pdf_content',
                    'text': chunk
                }
            }]
            
            index.upsert(vectors=vector)
            print(f"Chunk {chunk_idx + 1}/{len(text_chunks)} processed")
            
        print(f"Document {doc.doc_id} processed successfully")
    
    print("PDF documents imported successfully into Pinecone vector database")

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        metadata = x['metadata']
        content_type = metadata.get('type', 'unknown')
        
        if content_type == 'website_content':
            text = (
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Content: {metadata.get('content', 'N/A')}\n"
                f"Link: {metadata.get('link', 'N/A')}\n"
            )
        elif content_type == 'product':
            text = (
                f"Name: {metadata.get('name', 'N/A')}\n"
                f"Description: {metadata.get('description', 'N/A')}\n"
                f"Price: {metadata.get('price', 'N/A')}\n"
                f"Link: {metadata.get('link', 'N/A')}\n"
                f"Image Link : {metadata.get('image', 'N/A')}"
            )
        elif content_type == 'pdf_content':
            text = (
                f"File: {metadata.get('file_name', 'N/A')}\n"
                f"Content: {metadata.get('text', 'N/A')}\n"
            )
        else:
            # Fallback for unknown content types
            text = f"Content: {str(metadata)}\n"
            
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str

def query_pinecone(query: str, top_k = 5):
    #query pinecone and return list of records
    xq = embed_model.embed_documents([query])

    # initialize the vector store object
    xc = index.query(
        vector=xq[0], top_k=top_k, include_values=True,include_metadata=True
    )

    context_str = format_rag_contexts(xc["matches"])
    return context_str

# Entry point
# if __name__ == "__main__":
#     import_json_to_vector(f"{config.SOURCE}/post.json","website")
#     import_json_to_vector(f"{config.SOURCE}/products.json","product")
#     print(index.describe_index_stats())