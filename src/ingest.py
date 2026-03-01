import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def get_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        try:
            emb.embed_query(".")
            return emb
        except Exception as e:
            if "insufficient_quota" in str(e) and os.getenv("GOOGLE_API_KEY"):
                print("Quota OpenAI excedida. Usando Gemini como fallback...")
            else:
                raise

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"))

    raise ValueError("Nenhuma API key encontrada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env")


def ingest_pdf():
    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"PDF dividido em {len(chunks)} chunks")

    embeddings = get_embeddings()

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
    )

    print(f"{len(chunks)} chunks ingeridos com sucesso no banco de dados.")


if __name__ == "__main__":
    ingest_pdf()
