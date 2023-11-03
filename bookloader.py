# bookloader.py
import os
import glob
# Langchain imports
from langchain.document_loaders import (
    PyMuPDFLoader,
    UnstructuredEPubLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# #### CONFIG ####
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral-openorca"
CHROMADB_DIR = os.environ.get(
    'CHROMADB_DIR',
    './book_db'
)
SOURCE_DIRECTORY = os.environ.get(
    'SOURCE_DIRECTORY',
    'source_documents'
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".epub": (UnstructuredEPubLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
# ################


def main():
    """ Load data into ChromaDB from ebooks, text, or PDFs """
    # Setup embeddings
    oembed = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=0.75,
    )

    # Get all the file paths, load into list
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(SOURCE_DIRECTORY, f"**/*{ext}"), recursive=True)
        )

    # Loop through all_files and then embed and store into a DB as chunks
    # TODO Write a function to check if document is already in ChromaDB and skip import
    # TODO add a fancy loader animation thing for a console output
    for file_path in all_files:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            print(f"Working on: {file_path}")
            # Get the right loader
            loader_class, loader_args = LOADER_MAPPING[ext]
            # Set the loader up
            loader = loader_class(
                file_path,
                **loader_args
            )
            # Split up text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            # Chunk it up
            texts = text_splitter.split_documents(
                documents=loader.load()
            )
            # Load the text into the DB
            db = Chroma.from_documents(
                persist_directory=CHROMADB_DIR,
                documents=texts,
                embedding=oembed
            )
            # Make sure we keep the data persistent
            db.persist()
            print(f"Finished {file_path}, next.")

    print("FINISHED")


if __name__ == '__main__':
    main()
