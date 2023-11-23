# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from custom_processing import process_data, categorize_questions
from pyspark.sql import SparkSession

import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('question_form.html')


@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Server error: {e}")
    return "Internal Server Error", 500


@app.route('/submit_question', methods=['POST'])
def submit_question():
    # Extract question from the form
    questions = request.form.get('questions')

    # Initialize Spark session
    spark = SparkSession.builder.appName("LlamaOracle").getOrCreate()

    # Call a function to categorize questions (this function needs to be implemented)
    categorized_questions = categorize_questions(spark, [(questions, None)])

    print(type(categorized_questions))
    print(categorized_questions)

    questions_list = categorized_questions.collect()
    questions_dict = [row.asDict() for row in questions_list]


    """
    # Call the Spark processing function with the question
    answer_df = process_data(spark, categorized_questions)

    answers_list = answer_df.collect()
    answers_dict = [row.asDict() for row in answers_list]
    """

    # Terminate the Spark session
    spark.stop()

    # Return the answer
    return jsonify(questions_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
