# src/results_rag/vector_search.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def create_vector_index_from_pages(pages):
    """
    Create a TF-IDF vector index from page metadata.
    
    Args:
        pages: List of page metadata objects
        
    Returns:
        Tuple of (vectorizer, document_vectors, pages)
    """
    # Extract text content to vectorize
    documents = []
    for page in pages:
        # Combine page summary and table metadata
        table_text = " ".join([
            f"{table.get('table_caption', '')} {table.get('table_summary', '')}"
            for table in page.get('tables', [])
        ])
        document_text = f"{page.get('page_summary', '')} {table_text}"
        documents.append(document_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    document_vectors = vectorizer.fit_transform(documents)
    
    return vectorizer, document_vectors, pages

def search_relevant_pages(vectorizer, document_vectors, pages, query, top_k=5):
    """
    Search for relevant pages using TF-IDF and cosine similarity.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        document_vectors: Document vectors
        pages: List of page metadata objects
        query: Search query
        top_k: Number of top results to return
        
    Returns:
        List of relevant pages
    """
    # Transform query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get indices of top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return relevant pages
    return [pages[i] for i in top_indices if similarities[i] > 0.1]  # Set a threshold