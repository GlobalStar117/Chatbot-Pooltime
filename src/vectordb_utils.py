import json
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import csv
import uuid
from dotenv import load_dotenv
import config
import time



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
def import_json_to_vector(json_file_path, data_type="website"):
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
                    'link': item['link'],
                    'type': 'website_content'
                }
            else:  # product data
                text_to_embed = f"Name: {item['name']}\n{item['description']}"
                metadata = {
                    'name': item['name'],
                    'price': item['price'],
                    'description': item['description'],
                    'link': item['link'],
                    'type': 'product'
                }
            
            vector = [{
                'id': embedding_id,
                'values': embed_model.embed_documents([text_to_embed])[0],
                'metadata': metadata
            }]
            
            index.upsert(vectors=vector)
            print("Item processed successfully")

    print(f"JSON data imported successfully into pinecone vector database. Data type: {data_type}")

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        if x['metadata']['type'] == 'website_content':
            text = (
                f"Title: {x['metadata']['title']}\n"
                f"Link: {x['metadata']['link']}\n"
            )
        else:  # product data
            text = (
                f"Name: {x['metadata']['name']}\n"
                f"Description: {x['metadata']['description']}\n"
                f"Price: {x['metadata']['price']}\n"
                f"Link: {x['metadata']['link']}\n"
            )
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
    print(context_str)
    return context_str

# Entry point
# if __name__ == "__main__":
#     import_json_to_vector(f"{config.SOURCE}/post.json","website")
#     import_json_to_vector(f"{config.SOURCE}/products.json","product")
#     print(index.describe_index_stats())