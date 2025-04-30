import streamlit as st
from fashion_rag import FashionRAGAssistant

# Initialize FashionRAGAssistant with the collection name (e.g., "fashion_db")
assistant = FashionRAGAssistant(collection_name="fashion_db")

# Streamlit UI for input
st.title("Fashion RAG Assistant")
st.write("Ask your fashion-related query!")

# Input field for query
query = st.text_input("Enter your fashion query:")

# Button to trigger query processing
if st.button("Get Answer"):
    if query:
        # Retrieve ranked context for the query
        context_docs = assistant.retrieve_context_with_rank(query)

        # Generate an AI-powered answer using OpenAI
        answer = assistant.generate_answer(query, context_docs)

        # Display the AI answer
        st.write("AI Answer:")
        st.write(answer)
    else:
        st.error("Please enter a query to get an answer.")
