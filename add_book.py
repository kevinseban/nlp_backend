from flask import Blueprint, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import torch

# Initialize Blueprint for adding a book
add_book_bp = Blueprint('add_book', __name__)

# Load BART model and tokenizer for summarization
bart_model_name = 'facebook/bart-large-cnn'
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Load SentenceTransformer for embeddings
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# MongoDB connection
mongo_uri = "mongodb+srv://kevinseban03:pass123word@cluster0.urcpt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client['library']
collection = db['books']

# Function to generate summary using BART
def generate_summary(book_content):
    inputs = bart_tokenizer(book_content, truncation=True, padding='longest', return_tensors="pt", max_length=1024)
    bart_model.eval()
    summary_ids = bart_model.generate(
        inputs['input_ids'], 
        num_beams=4, 
        max_length=150, 
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# API endpoint to handle POST requests for book content
@add_book_bp.route('/', methods=['POST'])
def add_book():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if necessary fields are present in the request
        if 'title' not in data or 'author' not in data or 'book_content' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Title, author, and book content are required',
                'data': {}
            }), 400

        # Extract details from the request
        title = data['title']
        author = data['author']
        book_content = data['book_content']

        # Generate the summary for the book content
        summary = generate_summary(book_content)

        # Generate the embedding for the book content
        embedding = sentence_model.encode(book_content).tolist()

        # Insert the document into MongoDB
        book_document = {
            'title': title,
            'author': author,
            'summary': summary,
            'embedding': embedding
        }
        collection.insert_one(book_document)

        return jsonify({
            'status': 'success',
            'message': 'Summary generated and book data inserted successfully',
            'data': {
                'summary': summary
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': {}
        }), 500
