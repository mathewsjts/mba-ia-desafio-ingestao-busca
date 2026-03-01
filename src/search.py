import os
from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


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


def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-5-nano")
        try:
            llm.invoke(".")
            return llm
        except Exception as e:
            if "insufficient_quota" in str(e) and os.getenv("GOOGLE_API_KEY"):
                print("Quota OpenAI (LLM) excedida. Usando Gemini como fallback...")
            else:
                raise

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    raise ValueError("Nenhuma API key encontrada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env")


def search_prompt():
    try:
        embeddings = get_embeddings()
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            connection=DATABASE_URL,
        )
        llm = get_llm()

        def ask(q):
            results = vector_store.similarity_search_with_score(q, k=10)
            contexto = "\n\n".join([doc.page_content for doc, _ in results])
            prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=q)
            response = llm.invoke(prompt)

            return response.content

        return ask

    except Exception as e:
        print(f"Erro ao inicializar: {e}")
        return None
