# NLP Document Processing System

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/yourusername/nlp-document-processor/CI/CD%20Pipeline)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive document processing system with natural language processing capabilities for extracting insights from various document formats.

## 🌟 Features

- **Multi-format Document Parsing**: Support for PDF, DOCX, and TXT formats
- **Named Entity Recognition**: Extract people, organizations, locations, and dates with confidence scoring
- **Document Classification**: Automatically categorize documents by type and priority level
- **Document Summarization**: Generate concise summaries with configurable length and methods:
  - Abstractive (AI-generated new text)
  - Extractive (key sentence selection)
  - Hybrid (combination of both approaches)
- **Interactive UI**: User-friendly interface built with Gradio
- **RESTful API**: Robust API for integration with other systems
- **Visualization**: Visual representation of extracted information

## 🖼️ Demo

![Demo Screenshot](docs/images/demo-screenshot.png)

Try the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/yourusername/nlp-document-processor)

## 🧩 System Architecture

The system is structured into several key components:

```
nlp-document-processor/
├── preprocessing/ - Document parsing and text extraction
├── nlp/ - Core NLP capabilities
├── api/ - FastAPI endpoints
└── frontend/ - Gradio UI
```

![Architecture Diagram](docs/images/architecture.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nlp-document-processor.git
   cd nlp-document-processor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Running the Application

#### Start the API server:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

#### Start the Gradio UI:

```bash
python -m src.frontend.app
```

The UI will be available at `http://localhost:7860`

### Using Docker

```bash
# Build the Docker image
docker build -t nlp-document-processor .

# Run the container
docker run -p 8000:8000 -p 7860:7860 nlp-document-processor
```

## 📚 Usage

### API Endpoints

The RESTful API provides the following endpoints:

- `POST /api/process` - Process a document with all NLP capabilities
- `POST /api/extract-entities` - Extract named entities from a document
- `POST /api/classify-document` - Classify a document by type and priority
- `POST /api/summarize` - Generate a summary of a document

API Documentation (Swagger UI) is available at `http://localhost:8000/docs`

### Example API Request

```python
import requests

# Process a document
files = {'file': open('sample.pdf', 'rb')}
options = {
    'perform_ner': True,
    'perform_classification': True,
    'perform_summarization': True,
    'summary_length': 'medium',
    'summary_method': 'abstractive'
}

response = requests.post(
    'http://localhost:8000/api/process',
    files=files,
    data={'options': json.dumps(options)}
)

result = response.json()
print(f"Document type: {result['classification']['document_type']}")
print(f"Summary: {result['summary']['summary']}")
```

## 🧪 Testing

Run the tests using pytest:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_preprocessing.py
pytest tests/test_nlp.py
pytest tests/test_api.py

# With coverage report
pytest --cov=src
```

## 🔄 CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

- Runs test suite on each push
- Lints code with flake8
- Checks formatting with black
- Builds and deploys to Hugging Face Spaces on main branch

## 📋 Project Roadmap

- [x] Document parsing for PDF, DOCX, and TXT
- [x] Named Entity Recognition
- [x] Document Classification
- [x] Document Summarization
- [x] RESTful API
- [x] Gradio Interface
- [x] Test Coverage
- [ ] Support for more document formats (HTML, RTF)
- [ ] Improved visualization with interactive charts
- [ ] Custom NER models for domain-specific entities
- [ ] Question Answering capabilities

## 🔧 Technology Stack

- **Document Processing**: PyPDF2, pdfplumber, python-docx
- **NLP**: Transformers, NLTK, spaCy
- **API**: FastAPI, Pydantic
- **Frontend**: Gradio
- **Testing**: Pytest
- **Visualization**: Matplotlib, pandas
- **CI/CD**: GitHub Actions

## 🧰 Project Structure

```
nlp-document-processor/
├── src/
│   ├── preprocessing/   # Document parsing modules
│   ├── nlp/             # NLP processing modules
│   ├── api/             # FastAPI endpoints
│   └── frontend/        # Gradio interface
├── tests/               # Test suite
├── data/                # Example documents
│   ├── raw/             # Original documents
│   └── processed/       # Processed outputs
├── docs/                # Documentation
│   └── images/          # Screenshots and diagrams
├── .github/workflows/   # CI/CD configuration
├── requirements.txt     # Project dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md            # Project documentation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/nlp-document-processor](https://github.com/yourusername/nlp-document-processor)

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pretrained models
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Gradio](https://gradio.app/) for the UI components