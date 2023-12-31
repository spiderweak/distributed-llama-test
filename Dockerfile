# Base image with Python and Java (needed for PySpark)
FROM openjdk:8-jdk-slim as base

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl vim

RUN mkdir -p /app/models/
RUN mkdir -p /app/templates/

# Set the working directory
WORKDIR /app

# Copy only the requirements file, to cache the installed dependencies
COPY requirements.txt ./

COPY static /app/static

COPY templates/question_form.html /app/templates/question_form.html

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Give execution rights to the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Copy the rest of the application
COPY app.py /app/app.py
COPY custom_processing.py /app/custom_processing.py
COPY utils.py /app/utils.py

#COPY models/llama-2-7b-chat.Q6_K.gguf /app/models/llama-2-7b-chat.Q6_K.gguf
#COPY models/zephyr-7b-beta.Q6_K.gguf /app/models/zephyr-7b-beta.Q6_K.gguf
# Run with -v /path/to/local/models:/app/models to mount your local model folder to the app/models folder

# Define environment variable for the Llama model path
ENV LLAMA_MODEL_PATH="/app/models/zephyr-7b-beta.Q6_K.gguf"

# Set the script as the entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the command to run the application
CMD ["python3", "./app.py"]

