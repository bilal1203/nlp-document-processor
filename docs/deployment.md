# Deployment Guide

This document provides instructions for deploying the NLP Document Processing System to various platforms.

## Hugging Face Spaces Deployment

[Hugging Face Spaces](https://huggingface.co/spaces) provides an easy way to deploy Gradio and Streamlit apps for free, making it ideal for showcasing this project.

### Prerequisites

1. A [Hugging Face](https://huggingface.co/) account
2. Git installed on your local machine
3. The NLP Document Processing System codebase

### Steps to Deploy

1. **Create a new Space on Hugging Face**

   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click on "Create new Space"
   - Choose a name (e.g., "nlp-document-processor")
   - Select "Gradio" as the SDK
   - Choose a license (e.g., "MIT")
   - Click "Create Space"

2. **Prepare your repository for Hugging Face Spaces**

   Create a `requirements.txt` file specifically for Hugging Face (optimize for size due to platform constraints):

   ```
   numpy>=1.20.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   tqdm>=4.62.0
   PyPDF2>=2.10.0
   pdfplumber>=0.7.0
   python-docx>=0.8.11
   transformers>=4.18.0
   torch>=1.11.0
   nltk>=3.6.5
   fastapi>=0.78.0
   uvicorn>=0.17.6
   pydantic>=1.9.0
   python-multipart>=0.0.5
   gradio>=3.0.0
   chardet>=4.0.0
   rapidfuzz>=2.0.0
   ```

3. **Create app.py in the root directory**

   Create a simplified entry point that imports the Gradio app:

   ```python
   from src.frontend.app import create_gradio_interface

   # Create the Gradio interface
   app = create_gradio_interface()

   # For Hugging Face Spaces
   if __name__ == "__main__":
       app.launch()
   ```

4. **Add a .gitattributes file for Large File Storage**

   ```
   *.pt filter=lfs diff=lfs merge=lfs -text
   *.pth filter=lfs diff=lfs merge=lfs -text
   *.bin filter=lfs diff=lfs merge=lfs -text
   ```

5. **Initialize NLTK and download required data**

   Create a `download_nltk_data.py` file:

   ```python
   import nltk

   nltk.download('punkt')
   nltk.download('stopwords')
   ```

   Add this to your `app.py`:

   ```python
   # Download NLTK data at startup
   import nltk
   try:
       nltk.data.find('tokenizers/punkt')
       nltk.data.find('corpora/stopwords')
   except LookupError:
       nltk.download('punkt')
       nltk.download('stopwords')
   ```

6. **Create a Dockerfile for Hugging Face Spaces**

   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       python3-dev \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements first for better caching
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Download NLTK data
   COPY download_nltk_data.py .
   RUN python download_nltk_data.py

   # Copy the application
   COPY . .

   # Expose the port
   EXPOSE 7860

   # Run the application
   CMD ["python", "app.py"]
   ```

7. **Clone the Hugging Face Space repository and push your code**

   ```bash
   git clone https://huggingface.co/spaces/yourusername/nlp-document-processor
   cd nlp-document-processor
   # Copy your project files here
   git add .
   git commit -m "Initial deployment"
   git push
   ```

8. **Monitor deployment**

   After pushing your code, Hugging Face will automatically build and deploy your application. You can monitor the build progress on your Space's page.

### Configuration Options

- **Environment Variables**: You can set environment variables in the Hugging Face interface under the "Settings" tab.
- **Hardware**: Hugging Face Spaces offers different hardware tiers, including CPU and GPU options. For better performance with transformer models, consider upgrading to a GPU instance.
- **Persistent Storage**: To store uploaded files across restarts, use the `/data` directory which is persistent.

### Limitations

- **Compute Resources**: Free tier has limited compute resources, which may affect performance for large documents.
- **Memory Limits**: Be mindful of memory usage, especially when loading multiple transformer models.
- **Storage**: There is a file size limit for uploads, consider implementing size restrictions in your app.

## AWS Deployment

For production deployments with higher scalability, AWS provides several options.

### Option 1: AWS Elastic Beanstalk

1. **Prerequisites**
   - AWS account
   - AWS CLI installed and configured
   - EB CLI installed

2. **Create an Elastic Beanstalk application**
   ```bash
   eb init -p python-3.9 nlp-document-processor
   eb create nlp-document-processor-env
   ```

3. **Configure environment variables**
   ```bash
   eb setenv HOST=0.0.0.0 PORT=8080
   ```

4. **Deploy the application**
   ```bash
   eb deploy
   ```

### Option 2: Docker on AWS ECS

1. **Create an ECR repository**
   ```bash
   aws ecr create-repository --repository-name nlp-document-processor
   ```

2. **Build and push Docker image**
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
   docker build -t nlp-document-processor .
   docker tag nlp-document-processor:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/nlp-document-processor:latest
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/nlp-document-processor:latest
   ```

3. **Create an ECS cluster and service** using the AWS console or CLI.

## GCP Deployment

Google Cloud Platform offers Cloud Run for containerized applications.

1. **Set up gcloud CLI and authenticate**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build and push Docker image to Google Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/nlp-document-processor
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy nlp-document-processor --image gcr.io/YOUR_PROJECT_ID/nlp-document-processor --platform managed
   ```

## Azure Deployment

Microsoft Azure provides App Service for containerized applications.

1. **Create Azure Container Registry**
   ```bash
   az acr create --resource-group myResourceGroup --name myRegistry --sku Basic
   ```

2. **Build and push Docker image**
   ```bash
   az acr build --registry myRegistry --image nlp-document-processor:latest .
   ```

3. **Create Azure App Service**
   ```bash
   az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux
   az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name nlp-document-processor --deployment-container-image-name myRegistry.azurecr.io/nlp-document-processor:latest
   ```