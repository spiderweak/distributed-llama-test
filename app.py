# Import necessary libraries
from flask import Flask, render_template, jsonify
from custom_processing import process_data

import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



questions_data = [
    ("What are the days of the week? And where does their name come from?", "days_of_week"),
    ("What are the months of the year? And where does their name come from?", "months_of_year"),
    ("What are the four seasons? When is the longest day of the year? When is the shortest day of the year? When is the equinox?", "seasons")
]


def index():
    logger.info("Index page requested")
    return render_template('index.html', questions=questions_data)

@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Server error: {e}")
    return "Internal Server Error", 500

# This route will process the data and return the answers
@app.route('/get_answers', methods=['GET'])
def get_answers():
    answers = process_data()
    return jsonify(answers)

if __name__ == '__main__':
    app.run(debug=True)