import argparse
import os
import shutil
import uuid
import logging
from typing import List
from dotenv import load_dotenv
import yaml
from langchain.schema.document import Document
from get_embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

load_dotenv()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
PROCESSED_FILES_PATH = os.getenv("PROCESSED_FILES_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        logger.info("Clearing Database")
        clear_database()

    new_files = get_new_files()
    if not new_files:
        logger.info("No new files to process")
        return

    documents = load_documents(new_files)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    update_processed_files(new_files)

def get_new_files() -> List[str]:
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            processed_files = f.read().splitlines()
    else:
        processed_files = []

    new_files = [f for f in all_files if f not in processed_files]
    return new_files

def load_documents(new_files: List[str]) -> List[Document]:
    documents = []
    for filename in new_files:
        file_path = os.path.join(DATA_PATH, filename)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                documents.append(Document(page_content=text, metadata={"source": filename}))
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['text_splitting']['chunk_size'],
        chunk_overlap=config['text_splitting']['chunk_overlap'],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    embedding_function = get_embedding_function(OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logger.info(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        logger.info("No new documents to add")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks

def update_processed_files(new_files: List[str]):
    with open(PROCESSED_FILES_PATH, 'a') as f:
        for file in new_files:
            f.write(f"{file}\n")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(PROCESSED_FILES_PATH):
        os.remove(PROCESSED_FILES_PATH)

if __name__ == "__main__":
    main()