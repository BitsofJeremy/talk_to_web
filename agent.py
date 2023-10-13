from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain import hub

# #### CONFIG ####
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral-openorca"
CHROMADB_DIR = "./db"
# ################

# Setup the LLM
llm = Ollama(
    model=OLLAMA_MODEL,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Setup embeddings
oembed = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model=OLLAMA_MODEL,
    temperature=1,
)

# Setup the vector store
vectorstore = Chroma(
    persist_directory=CHROMADB_DIR,
    embedding_function=oembed
)

# Setup the QA chain
PROMPT = hub.pull("rlm/rag-prompt")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)


def main():
    """ Send User query to the QA Chain """
    while True:
        user_input = input("query: ")
        print()
        result = qa_chain({"query": user_input})
        print()


if __name__ == "__main__":
    main()
