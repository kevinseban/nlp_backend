from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# Initialize the Blueprint for search_book
search_book_bp = Blueprint('search_book', __name__)

# MongoDB connection string
mongo_uri = "mongodb+srv://kevinseban03:pass123word@cluster0.urcpt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client['library']
collection = db['books']

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Download necessary NLTK resources (only once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Function to expand query with synonyms using WordNet
def expand_query(query):
    tokens = query.lower().split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    expanded_query = set(filtered_tokens)

    # Add synonyms from WordNet to the query
    for token in filtered_tokens:
        synonyms = set()
        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        expanded_query.update(synonyms)
    
    return ' '.join(expanded_query)

# Function to perform semantic search with query expansion
def search_books(query):
    expanded_query = expand_query(query)
    query_embedding = model.encode([expanded_query])
    query_embedding = np.array(query_embedding)

    books = collection.find()
    similarities = []

    for book in books:
        book_embedding = book['embedding']
        book_embedding = np.array(book_embedding)

        query_embedding = query_embedding.reshape(1, -1)
        book_embedding = book_embedding.reshape(1, -1)

        similarity = cosine_similarity(query_embedding, book_embedding)[0][0]
        
        if similarity >= 0.2:
            similarities.append((similarity, book))

    similarities.sort(key=lambda x: x[0], reverse=True)
    
    results = [{
        'title': book['title'],
        'author': book['author'],
        'summary': book['summary'],
        'similarity': similarity
    } for similarity, book in similarities[:5]]

    return results

# Define the search API route
@search_book_bp.route('/', methods=['GET'])
def search():
    query = request.args.get('query')
    
    if not query:
        return jsonify({"status": "error", "message": "No query provided"}), 400

    try:
        results = search_books(query)
        if not results:
            return jsonify({
                "status": "success",
                "message": "No books found matching the query",
                "data": []
            }), 200
        return jsonify({
            "status": "success",
            "message": "Books found successfully",
            "data": results
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500
