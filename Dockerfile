# Base image with Python and Java (needed for PySpark)
FROM openjdk:8-jdk-slim as base

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl vim

RUN mkdir -p /app/models/
RUN mkdir -p /app/templates

# Set the working directory
WORKDIR /app

# Copy only the requirements file, to cache the installed dependencies
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY index.html /app/templates/index.html

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Give execution rights to the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Copy the rest of the application
COPY llama2_test.py /app/llama2_test.py

COPY llama-2-7b-chat.Q6_K.gguf /app/models/llama-2-7b-chat.Q6_K.gguf

# Define environment variable for the Llama model path
ENV LLAMA_MODEL_PATH="/app/models/llama-2-7b-chat.Q6_K.gguf"

EXPOSE 5000

# Set the script as the entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the command to run the application
CMD ["python3", "./llama2_test.py"]
