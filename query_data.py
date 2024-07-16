import os
from dotenv import load_dotenv
import yaml
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding import get_embedding_function
import time
import logging
import colorlog

load_dotenv()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Configure colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

CHROMA_PATH = os.getenv("CHROMA_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
Answer the question of the user
Question: {question}
---
context:
{context}
---
"""

def main():
    logger.debug("🚀 Starting main function.")

    embedding_function = get_embedding_function(OPENAI_API_KEY)
    logger.debug(f"🧩 Embedding function created: {embedding_function}")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    logger.debug(f"📂 Chroma DB initialized at path: {CHROMA_PATH}")

    model = Ollama(model=config['llm']['model'])
    logger.debug(f"🤖 {config['llm']['model']} model initialized.")

    try:
        while True:
            query_text = input("Write your question and hit enter (or 'exit' to ... exit! ): ")
            if query_text.lower() == "exit":
                logger.debug("🔚 Exit command received. Exiting loop.")
                break

            logger.debug(f"📝 Received query: {query_text}")

            start_time = time.time()
            response_text = query_rag(query_text, db, model, config['llm']['temperature'])
            end_time = time.time()

            logger.debug(f"📜 Query response: {response_text}")
            logger.debug(f"⏱️ Time taken for query: {end_time - start_time:.2f} seconds")

            print(f"{response_text}\nTime taken: {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        logger.info("Received interrupt. Shutting down gracefully.")
    finally:
        # Perform any necessary cleanup here
        logger.info("Shutting down. Goodbye!")

def query_rag(query_text: str, db: Chroma, model: Ollama, temperature: float) -> str:
    logger.debug("🔍 Starting query_rag function.")

    results = db.similarity_search_with_score(query_text, k=config['similarity_search']['k'])
    logger.debug(f"🔗 Similarity search results: {results}")

    context_texts = []
    for doc, score in results:
        logger.debug(f"📄 Document ID: {doc.metadata.get('id', None)}")
        logger.debug(f"🔢 Document score: {score}")
        logger.debug(f"📃 Document content: {doc.page_content}")
        context_texts.append(doc.page_content)

    context_text = "\n\n---\n\n".join(context_texts)
    logger.debug(f"🧩 Context text: {context_text}")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    logger.debug(f"✏️ Generated prompt: {prompt}")

    response_text = model.invoke(prompt, temperature=temperature)
    logger.debug(f"💬 Model response: {response_text}")

    answer = response_text.split('---')[-1].strip()
    logger.debug(f"📝 Extracted answer: {answer}")

    if "Did not find an answer" in answer:
        answer = "Did not find any answer. Change your question. "

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    logger.debug(f"🔗 Sources: {sources}")

    formatted_response = f"{answer}\nSources: {sources}"
    logger.debug(f"📝 Formatted response: {formatted_response}")

    return formatted_response

if __name__ == "__main__":
    main()