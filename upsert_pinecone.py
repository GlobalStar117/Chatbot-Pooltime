from dotenv import load_dotenv
import os
from config import SOURCE
from src.vectordb_utils import import_json_to_vector, import_pdfs_to_vector, index

# Load environment variables at the start
load_dotenv()

def truncate_metadata(metadata, max_bytes=40000):
    """Truncate metadata to stay within Pinecone's limits"""
    total_bytes = 0
    truncated_metadata = {}
    
    for key, value in metadata.items():
        # Convert value to string if it isn't already
        value_str = str(value)
        value_bytes = len(value_str.encode('utf-8'))
        
        if total_bytes + value_bytes <= max_bytes:
            truncated_metadata[key] = value
            total_bytes += value_bytes
        else:
            # If it's a text field, truncate it
            if isinstance(value, str):
                remaining_bytes = max_bytes - total_bytes
                truncated_value = value_str.encode('utf-8')[:remaining_bytes].decode('utf-8', errors='ignore')
                truncated_metadata[key] = truncated_value
            break
            
    return truncated_metadata

def main():
    try:
        # Import different types of data into vector database
        import_json_to_vector(os.path.join(SOURCE, "pages.json"), "website", metadata_processor=truncate_metadata)
        import_json_to_vector(os.path.join(SOURCE, "posts.json"), "website")
        import_json_to_vector(os.path.join(SOURCE, "products.json"), "product")
        print(os.path.join(SOURCE))
        # import_pdfs_to_vector(os.path.join(SOURCE))
        # Print index statistics
        print("Vector Database Statistics:")
        print(index.describe_index_stats())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()