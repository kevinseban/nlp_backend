from flask import Flask, Response, request
from flask_cors import CORS
from add_book import add_book_bp
from search_book import search_book_bp

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        # Returning an empty response for OPTIONS requests (CORS preflight)
        return Response(status=200)

# Register blueprints for add_book and search_book
app.register_blueprint(add_book_bp, url_prefix='/add')
app.register_blueprint(search_book_bp, url_prefix='/search')

# Home route (optional)
@app.route('/')
def home():
    return "Welcome to the Book Management App!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
