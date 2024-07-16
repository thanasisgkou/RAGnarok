# RAGnarok

RAGnarok is a Retrieval-Augmented Generation (RAG) system that processes PDF documents, stores their content in a vector database, and allows users to query the information using natural language.

## Project Structure

```
RAGnarok/
│
├── chroma/                 # Vector database storage
├── data/                   # Directory for storing input PDF files
│   └── processed_files.txt # List of processed PDF files
├── .env                    # Environment variables
├── config.yml              # Configuration settings
├── get_embedding.py        # Embedding functionality
├── populate_database.py    # Script to process PDFs and populate the database
├── query_data.py           # Script to run the query interface
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Prerequisites

Before setting up RAGnarok, you need to install Ollama and download the llama3 model:

1. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/).

2. Once Ollama is installed, download the llama3 model by running:
   ```
   ollama pull llama3
   ```

This step is crucial as RAGnarok uses the llama3 model for generating responses.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/RAGnarok.git
   cd RAGnarok
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Adjust the `config.yml` file if needed to customize settings.

## Usage

### Populating the Database

1. Place your PDF files in the `data/` directory.

2. Run the database population script:
   ```
   python populate_database.py
   ```

   This will process the PDF files, extract their text, split it into chunks, create embeddings, and store them in the Chroma vector database.

### Querying the Data

1. To start the query interface, run:
   ```
   python query_data.py
   ```

2. Enter your questions when prompted. The system will retrieve relevant information from the processed documents and generate an answer using the llama3 model.

3. Type 'exit' to quit the query interface.

## Configuration

You can adjust various settings in the `config.yml` file, including:
- Embedding model parameters
- Text splitting parameters
- Similarity search settings
- LLM (Language Model) settings

## Contributing

Contributions to RAGnarok are welcome! Please feel free to submit a Pull Request.

## License

[Buy Thanasis a beer]
