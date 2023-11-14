# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from custom_processing import process_data
from pyspark.sql import SparkSession

import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('question_form.html')


@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Server error: {e}")
    return "Internal Server Error", 500


@app.route('/submit_question', methods=['POST'])
def submit_question():
    # Extract question and category from the form
    question = request.form.get('question')
    category = request.form.get('category')

    # Initialize Spark session
    spark = SparkSession.builder.appName("LlamaOracle").getOrCreate()

    # Call the Spark processing function with the question and category
    answer_df = process_data(spark, [(question, category)])

    # Convert the result to a dictionary (or any other format you prefer)
    answer = answer_df.toPandas().to_dict(orient='records')

    # Terminate the Spark session
    spark.stop()

    # Return the answer
    return jsonify(answer)


if __name__ == '__main__':
    app.run(debug=True)