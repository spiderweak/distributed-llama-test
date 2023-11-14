# Import necessary libraries
import pandas as pd
import re
from llama_cpp import Llama
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


# Initialize Spark session
spark = SparkSession.builder.appName("LlamaSummarizer").getOrCreate()

schema = StructType([
        StructField("summary", StringType(), True),
        StructField("chapter", IntegerType(), True)
    ])

# this is the function applied per-group by Spark
# the df passed is a *Pandas* dataframe!
def llama2_summarize(df):
    llm = Llama(model_path="/home/humanitas/models/llama-2-7b-chat.Q4_K_M.gguf",
              n_ctx=8192,
              n_batch=512)

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
    
    # create prompt
    chapter_text = df.iloc[0]['text']
    chapter_num = df.iloc[0]['chapter']
    prompt = 'Summarize the following novel chapter in a single sentence (less than 100 words):' + chapter_text
    prompt = template.replace('INSERT_PROMPT_HERE', prompt)
    
    output = llm(prompt, 
                 max_tokens=-1, 
                 echo=False, 
                 temperature=0.2, 
                 top_p=0.1)

    return pd.DataFrame({'summary': [output['choices'][0]['text']], 
                     'chapter':[int(chapter_num)]})

summarize_udf = llama2_summarize

# read book, remove header/footer
text = open('./data/war_and_peace.txt', 'r').read()
text = text.split('PROJECT GUTENBERG EBOOK WAR AND PEACE')[1]

# get list of chapter strings
chapter_list = [x for x in re.split('CHAPTER .+', text) if len(x) > 100]

# create Spark dataframe, show it
df = spark.createDataFrame(pd.DataFrame({'text':chapter_list,
                                         'chapter':range(1,len(chapter_list)+1)}))

# Now you can apply the UDF to the DataFrame
summaries = (df
             .limit(3)  # If you are limiting to 1, no need for groupBy. If summarizing all, remove this line.
             .groupby('chapter')
             .applyInPandas(summarize_udf, schema=schema)
             )

# Show the summaries
summaries.show(vertical=True, truncate=False)
