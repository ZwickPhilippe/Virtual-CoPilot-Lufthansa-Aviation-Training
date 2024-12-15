# Lufthansa Virtual CoPilot - how to run?

If you have already completed steps 1-6 before, then only repeat steps 3, 4, 6.

## Step 1: Hugging Face token

Get a Hugging Face token (https://huggingface.co/docs/hub/security-tokens) of token type 'read' and put it into the file HF_token.txt.

## Step 2: Install Docker 

Download and install Docker Desktop (https://docker.com)

## Step 3: Run Docker Desktop on your machine

Simply open the Docker Desktop application.

## Step 4: Open Terminal

Open this file's parent folder POC_Chatbot in a terminal such that you can execute the commands below.

## Step 5: Build the docker image

```docker build -t lufthansa-virtual-copilot .```

## Step 6: Start the docker container

```docker run -p 8501:8501 lufthansa-virtual-copilot```

## Access the application

Go to http://localhost:8501 in your browser to use Virtual CoPilot.