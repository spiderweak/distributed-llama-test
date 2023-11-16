# Import necessary libraries
from typing import Any, List, Tuple
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
            question_category = row['category']
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
            answers.append((answer_text, question_category))

        # Return the DataFrame with answers.
        return pd.DataFrame(
            answers,
            columns=['answer', 'category']
        )

def process_data(spark_session: SparkSession, questions_data: List[Tuple[str, str]]) -> pd.DataFrame:

    questions_df = spark_session.createDataFrame(data=questions_data, schema=question_schema)

    answers_df = llama2_answer_questions(questions_df.toPandas())

    return answers_df


questions_data = [
    ("What are the days of the week? And where does their name come from?", "days_of_week"),
    ("What are the months of the year? And where does their name come from?", "months_of_year"),
    ("What are the four seasons? When is the longest day of the year? When is the shortest day of the year? When is the equinox?", "seasons")
]

# Example usage
if __name__ == "__main__":
    spark = SparkSession.builder.appName("LlamaOracle").getOrCreate()
    answers = process_data(spark, questions_data)
    print(answers)