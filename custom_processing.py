# Import necessary libraries
from typing import Any, List, Tuple, Optional
import pandas as pd
import re
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import logging

# Define schemas
question_schema = StructType([
    StructField("question", StringType(), True),
    StructField("category", StringType(), True)
])

answer_schema = StructType([
    StructField("answer", StringType(), True),
    StructField("category", StringType(), True)
])

def llama2_answer_questions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply function for Spark to call Llama model and get answers for questions.

        Args:
            df: Pandas DataFrame with columns 'question' and 'category'.

        Returns:
            DataFrame with the Llama model's responses.
        """
        from llama_cpp import Llama
        import os

        # Configuration: Model path should be an environment variable.
        model_path = os.getenv("LLAMA_MODEL_PATH")

        llm = Llama(model_path=model_path, n_ctx=8192, n_batch=512)

        # template for this model version, see:
        # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML#prompt-template-llama-2-chat
        template = """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. 
        Always answer as helpfully as possible, while being safe.  
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
        Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
        If you don't know the answer to a question, please don't share false information.
        <</SYS>>

        {INSERT_PROMPT_HERE} [/INST]
        """

        # Process each of the provided rows
        answers = []
        for index, row in df.iterrows():
            question_text = row['question']
            category = row['category']
            prompt = 'Please answer the following questions in a single sentence (less than 100 words): ' + question_text
            prompt = template.replace('{INSERT_PROMPT_HERE}', prompt)

            try:
                output = llm(prompt, max_tokens=2048, echo=False)
                answer_text = output['choices'][0]['text']
                print(answer_text)
            except Exception as e:
                # Log the error and return an empty DataFrame or handle it as needed.
                logging.error(f"Error when calling Llama model: {e}")
                answer_text = "Error in processing"
            # Append the answer and category to the answers list
            answers.append((answer_text, category))

        # Return the DataFrame with answers.
        return pd.DataFrame(
            answers,
            columns=['answer', 'category']
        )

def process_data(spark_session: SparkSession, questions_data: List[Tuple[str, Optional[str]]]):

    questions_df = spark_session.createDataFrame(data=questions_data, schema=question_schema)

    answers = (questions_df
                .groupby('category')
                .applyInPandas(llama2_answer_questions, schema=answer_schema)
                )

    return answers

def categorize_questions(spark_session: SparkSession, questions_data: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:

    questions_df = spark_session.createDataFrame(data=questions_data, schema=question_schema)

    answers = (questions_df
                .groupby('category')
                .applyInPandas(llama_2_categorize_questions, schema=question_schema)
                )

    return answers

def llama_2_categorize_questions(df: pd.DataFrame) -> List[Tuple[str, Optional[str]]]:

    from llama_cpp import Llama
    import os

    # Configuration: Model path should be an environment variable.
    model_path = os.getenv("LLAMA_MODEL_PATH")

    llm = Llama(model_path=model_path, n_ctx=8192, n_batch=512)

    # template for this model version, see:
    # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML#prompt-template-llama-2-chat
    template = """
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe.

    You will be asked a set of questions, I want you to split all 
    the chunck of text into categories, and return the questions
    list in a list of dictionary format as follow : 

    [
        {
            question: <QUESTION_1_CONTENT>,
            category: <QUESTION_1_CATEGORY>
        },
        {
            question: <QUESTION_2_CONTENT>,
            category: <QUESTION_2_CATEGORY>
        },
        {
            question: <QUESTION_3_CONTENT>,
            category: <QUESTION_3_CATEGORY>
        },
        ...
    ]

    You are free to choose the category of your choice based on the question asked, but please stick to a 1 word category.
    <</SYS>>

    {INSERT_PROMPT_HERE} [/INST]
    """

    # Process each of the provided rows
    answers = []
    for index, row in df.iterrows():
        question_text = row['question']
        category = row['category']
        prompt = 'Here are the questions: ' + question_text
        prompt = template.replace('{INSERT_PROMPT_HERE}', prompt)

        try:
            output = llm(prompt, max_tokens=2048, echo=False)
            answer_text = output['choices'][0]['text']
            print(answer_text)
        except Exception as e:
            # Log the error and return an empty DataFrame or handle it as needed.
            logging.error(f"Error when calling Llama model: {e}")
            answer_text = "Error in processing"
        # Append the answer and category to the answers list
        output=parse_questions(answer_text)
        answers.append(output)

    # Return the DataFrame with answers.
    return answers

import re

def parse_questions(text):
    category_pattern = re.compile(r'^\s*[Cc]ategory:\s*(.+)\s*$')
    question_pattern = re.compile(r'^\s*[Qq]uestion(?: \d+)?:\s*(.+)\s*$')

    parsed_data = []
    last_category = None
    last_question = None

    for line in text.split('\n'):
        category_match = category_pattern.match(line)
        question_match = question_pattern.match(line)

        if category_match:
            # If a new category is found and a question is already stored, pair them
            if last_question:
                parsed_data.append((last_question, category_match.group(1).strip()))
                last_question = None  # Reset last_question
            else:
                last_category = category_match.group(1).strip()

        elif question_match:
            # If a new question is found and a category is already stored, pair them
            if last_category:
                parsed_data.append((question_match.group(1).strip(), last_category))
                last_category = None  # Reset last_category
            else:
                last_question = question_match.group(1).strip()

    return parsed_data


questions_data = [
    ("What are the days of the week? And where does their name come from?", None),
    ("What are the months of the year? And where does their name come from?", None),
    ("What are the four seasons? When is the longest day of the year? When is the shortest day of the year? When is the equinox?", None)
]

# Example usage
if __name__ == "__main__":
    spark = SparkSession.builder.appName("LlamaOracle").getOrCreate()
    answers = process_data(spark, questions_data)
    print(answers)