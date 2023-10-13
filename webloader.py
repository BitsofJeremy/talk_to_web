# webloader.py
import argparse
import sys
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# #### CONFIG ####
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral-openorca"
CHROMADB_DIR = "./db"
TEST_QUESTION = "FINISH the last paragraph: This is our world now..."
# ################


def main(**kwargs):
    """ Load data into ChromaDB from the web """
    # Get URL from args
    website_url = kwargs.get('website_url')

    # Setup embeddings
    oembed = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=1,
    )

    # Data Loader Setup
    loader = WebBaseLoader(website_url)
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(data)

    # Embed and store into a directory
    vectorstore = Chroma.from_documents(
        persist_directory=CHROMADB_DIR,
        documents=all_splits,
        embedding=oembed
    )

    # Retrieval test
    docs = vectorstore.similarity_search(TEST_QUESTION)
    print(docs)
    print(len(docs))
    print("FINISHED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-www',
        action='store',
        dest='website_url',
        help='Enter the website_url',
        required=True
    )
    args = parser.parse_args()
    # Convert the argparse.Namespace to a dictionary: vars(args)
    arg_dict = vars(args)
    # pass dictionary to main
    main(**arg_dict)
    sys.exit(0)
