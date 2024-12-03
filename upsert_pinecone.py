from dotenv import load_dotenv
import os
from config import SOURCE
from chatbot.src.vectordb_utils import import_json_to_vector, index

# Load environment variables at the start
load_dotenv()

def main():
    try:
        # Import different types of data into vector database
        import_json_to_vector(os.path.join(SOURCE, "post.json"), "website")
        import_json_to_vector(os.path.join(SOURCE, "products.json"), "product")
        
        # Print index statistics
        print("Vector Database Statistics:")
        print(index.describe_index_stats())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()