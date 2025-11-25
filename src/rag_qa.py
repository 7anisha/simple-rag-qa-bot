from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

PERSIST_DIR = str((__file__).resolve().parents[1] / "data" / "chroma_store")

def get_qa_chain(openai_api_key=None, model="gpt-4o-mini"):
    os.environ["OPENAI_API_KEY"] = openai_api_key or os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0.0, model=model)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="map_reduce")  # or "stuff"
    return qa

def answer(query, api_key=None):
    qa = get_qa_chain(openai_api_key=api_key)
    return qa.run(query)

if __name__ == "__main__":
    print(answer("Which movie features time travel and dinosaurs?", api_key=os.getenv("OPENAI_API_KEY")))
