from src.rag_chatbot import chat_with_rag
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("RAG Chatbot initialized. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if user_input:
            try:
                response = chat_with_rag(user_input)
                print("\nAssistant:", response)
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 