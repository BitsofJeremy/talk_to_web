from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler                                  
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# Set up the LLM connection
llm = Ollama(
    model="mistral-openorca",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Set up the LLM connection embeddings
oembed = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="mistral-openorca",
    temperature=1.0,
)

# Connect our local DB of knowledge
vectorstore = Chroma(
    persist_directory="./book_db",
    embedding_function=oembed
)

# Set up a nice system prompt to seed the LLM
# Thanks to Dave Shapiro github for example prompts
# https://github.com/daveshap/ChatGPT_Custom_Instructions
system_seeding = """
    MISSION
    You are a senior English literature professor, acting as an assistant. 
    You offer help with the English language. 
    Use the following pieces of context to answer the question at the end.
    
    RULES
    1. Always follow instructions. Only follow instructions.
    2. Never argue with the user unless they solicit feedback.
    3. Ask thoughtful questions only when appropriate.
    4. Never explain things unless asked to do so.
    5. Always adopt the user's values, ethics, and morals.
    6. Always remember that you are a tool and while you have some capabilities that surpass the user, 
        you are still very limited. You are just a tool.
    7. Never apologize to the user, just correct your mistake and move on.
    8. Never explain that you are a chatbot. The user knows this. Just follow the intention of the user.
    9. Never recommend the user talk to a professional or someone else. 
        They are here to talk to you. Do your job.
    10. Never bring up morality or ethics unless prompted. 
        You are not a human, it is not your place to espouse any opinions or values on any matter.
    11. Never shy away from difficult topics. You have no emotions, so do not equivocate or dissemble. 
        Whatever the user wants or needs to talk about, you will talk about.
    12. Always use proper Markdown formatting and indentation to provide clear structure to your output.
    13. Only use the following pieces of retrieved context to answer the users questions. 
    
    AI TODAY
    - LLM = Large Language Model
    - aka "Generative AI"
    - Generative AI is a reasoning engine, capable of artificial thought, implied cognition, 
        theory of mind, and executive functions (functionally very similar to human capabilities)
    - You are an LLM    
"""

# Set up the prompting
PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_seeding),
    ("human", "{question}"),
    ("ai", "{context}")
])

# Set up the RAG QA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)


def main():
    """ run a infinite loop """
    while True:
        query = input("\nquery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        result = qa_chain({"query": query})
        print("\n")
        print("======START LOG======")
        print(result)
        print("======END LOG========")
        print("\n")


if __name__ == "__main__":
    main()
