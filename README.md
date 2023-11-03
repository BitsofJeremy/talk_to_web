# talk_to_web

An example in Python for talking to webpages, and documents using Ollama, ChromaDB, and Langchain.

- UPDATE: Added two scripts to talk to books, or text.

## Install

Pull down the repo, create a virtualenv, activate and install requirements

```bash
git clone https://github.com/ephergent/talk_to_web.git
cd talk_to_web
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup Ollama

Download and install for your OS:

https://ollama.ai/download


Pull down a model to use, Mistral OpenOrca is pretty nice.

_The code is set for mistral-openorca_

```bash
ollama pull mistral-openorca
```

Ollama's API runs on `http://localhost:11434` by default. 

Open your browser to the local URL and verify: `Ollama is running`

You can manually start it with `ollama serv` in the terminal.

## Document Ingest

Run the `webloader.py` script with the URL you'd like to ingest.

```bash
python webloader.py -www "https://en.wikisource.org/wiki/The_Hacker_Manifesto"
```

## Run the agent:

```bash
 python agent.py 
 ```

Talk to your web page.

```bash
query: recite the last paragraph of the Hackers Manifesto

```

```bash
  This is our world now... the world of the electron and the switch, the beauty of the baud. We make use of a service already existing without paying for what could be dirt-cheap if it wasn't run by profiteering gluttons, and you call us criminals. We explore... and you call us criminals. We seek after knowledge... and you call us criminals. We exist without skin color, without nationality, without religious bias... and you call us criminals. You build atomic bombs, you wage wars, you murder, cheat, and lie to us and try to make us believe it's for our own good, yet we're the criminals. Yes, I am a criminal. My crime is that of curiosity. My crime is that of judging people by what they say and think, not what they look like. My crime is that of outsmarting you, something that you will never forgive me for. I am a hacker, and this is my manifesto. You may stop this individual, but you can't stop us all... after all, we're all alike.
```

# Book loader and agent scripts:

Drop a PDF, Epub, or Text in the `source_documents` directory.

Run `bookloader.py`, it will ingest the text into the ChromaDB

Run `book_agent.py` to connect to a chat agent (Ollama), and talk to the text.
- This uses `mistral-openorca` as the LLM.
- Run `ollama pull mistral-openorca` to get the LLM as stated above.


#### Free book used as example from Project Gutenberg:

Frankenstein; Or, The Modern Prometheus by Mary Wollstonecraft Shelley

https://www.gutenberg.org/ebooks/41445


# References used:

https://github.com/jmorganca/ollama/tree/main/examples/langchain-python-rag-document

https://python.langchain.com/docs/use_cases/question_answering/

https://python.langchain.com/docs/integrations/vectorstores/chroma

https://python.langchain.com/docs/integrations/chat/ollama





