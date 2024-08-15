from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from data import docs2

# loading environment variables from .env file
load_dotenv()

# Setting embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Testing if embedding works
vector = embeddings.embed_query('Testing the embedding model')
# print(vector)

# Connection string and collection name of the postgres db
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"

# Setting up vector store and passing values to be embedded and stored in the database
vector_store = PGVector.from_documents(
    documents=docs2,
    embedding=embeddings,
    connection=connection,
    collection_name=collection_name,
    use_jsonb=True,
)

# Query for the llm
query = "Tell me something about the exhibit"

# similar = vector_store.similarity_search_with_score(query=query,k=1)

# for doc, score in similar:
#     print(f"SIM: {score: 3f} {doc.page_content}")


# setting up retriever to test
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})

# Printing out what is getting fetched from postgres 
retrieved_docs = retriever.invoke(query)
for retrieved_doc in retrieved_docs:
    print(retrieved_doc.page_content)

# setting up llm
llm = ChatOpenAI(model="gpt-4o-mini")


# prompt string
prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 
    Context: {context} 
    Answer:
"""

# Creating prompt template from the above prompt string 
prompt_template = PromptTemplate.from_template(prompt)

# function to join data fetched from postgres into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# setting up retriever for LCEL chain
retriever2 = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})

# chain to get a question => call the postgres retriever => send the array of documents to format_docs function
#  => fill the required data into the prompt template => send everything to llm => retrieve output from llm as string
rag_chain = (
    {"question": RunnablePassthrough(), "context": retriever2 | format_docs}
    | prompt_template
    | llm
    | StrOutputParser()
)

# calling the chain using invoke method
response = rag_chain.invoke(query)
print(response)

# calling the chain using stream method
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)

