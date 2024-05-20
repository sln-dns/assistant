from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import sys
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()


openai_api_key=os.getenv("OPENAI_API_KEY")
local_model = "gpt-4-turbo"
embedding_model = "text-embedding-3-large"


def main():
    while True:
        local_path = input("Enter the path to the PDF file (or 'quit' to exit): ")

        if local_path.lower() == "quit":
            print("Exiting the program...")
            sys.exit(0)

        try:
            loader = PyPDFLoader(file_path=local_path)
            data = loader.load()
            break  
        except Exception as e:
            print("Error loading PDF file:", e)
            print("Please try again or enter 'quit' to exit.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)

    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(model=embedding_model),
        collection_name="local-rag"
    )

    # Define LLM
    
    llm = ChatOpenAI(model=local_model, temperature=0, api_key=openai_api_key)

    # Prompt template for queries
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Ты - ассистент языковой модели искусственного интеллекта. Твоя задача - сгенерировать пять
        различных версий заданного вопроса пользователя для извлечения соответствующих документов из
        векторной базы данных. Генерируя различные точки зрения на вопрос пользователя, твоя
        цель - помочь пользователю преодолеть некоторые ограничения поиска по сходству на основе расстояния
        поиска по сходству. Приведи эти альтернативные вопросы, разделенные новыми строками.
        Оригинальный вопрос: {question}""",
    )

    # Create retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Ответьте на вопрос, основываясь ТОЛЬКО на следующем контексте:
    {context}
    Question: {question}"""

    # Create prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Define chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User input loop
    while True:
        user_input = input("Enter your question (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            # Delete all collections in the db
            vector_db.delete_collection()
            break
        result = chain.invoke(user_input)
        print("Answer:", result)

if __name__ == "__main__":
    main()