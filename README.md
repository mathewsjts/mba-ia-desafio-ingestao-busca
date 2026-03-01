# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de **ingestão e busca semântica** sobre documentos PDF utilizando LangChain, PostgreSQL com pgVector e modelos de linguagem (OpenAI ou Google Gemini).

## Visão Geral

O projeto implementa um pipeline RAG (Retrieval-Augmented Generation):

1. **Ingestão** — lê um PDF, divide em chunks, gera embeddings e armazena os vetores no PostgreSQL.
2. **Busca** — ao receber uma pergunta, vetoriza a query, recupera os trechos mais relevantes do banco e envia ao LLM para gerar uma resposta baseada exclusivamente no conteúdo do documento.

## Pré-requisitos

- Python 3.13 (Python 3.14+ ainda não é suportado pelas dependências)
- Docker e Docker Compose
- Chave de API da **OpenAI** (`OPENAI_API_KEY`) **ou** da **Google** (`GOOGLE_API_KEY`)

## Configuração

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone git@github.com:mathewsjts/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca

python3.13 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# Escolha um dos dois providers (OpenAI tem prioridade se ambos estiverem preenchidos)

OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# GOOGLE_API_KEY=AIza...
# GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001

# Banco de dados (padrão do docker-compose.yml)
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Nome da coleção de vetores no banco
PG_VECTOR_COLLECTION_NAME=documents

# Caminho para o PDF a ser ingerido
PDF_PATH=document.pdf
```

## Execução

### 1. Subir o banco de dados

```bash
docker-compose up -d
```

Aguarde o container `postgres_rag` ficar saudável. A extensão `vector` é habilitada automaticamente pelo serviço `bootstrap_vector_ext`.

### 2. Ingerir o PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Carregando PDF: document.pdf
PDF dividido em 42 chunks
42 chunks ingeridos com sucesso no banco de dados.
```

### 3. Iniciar o chat

```bash
python src/chat.py
```

## Exemplo de uso

```
Chat iniciado. Digite 'sair' para encerrar.

Faça sua pergunta:
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Faça sua pergunta:
PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---

Faça sua pergunta:
PERGUNTA: sair
Encerrando o chat.
```

## Estrutura do projeto

```
├── docker-compose.yml          # PostgreSQL + pgVector
├── requirements.txt            # Dependências Python
├── .env.example                # Template de variáveis de ambiente
├── src/
│   ├── ingest.py               # Ingestão do PDF no banco vetorial
│   ├── search.py               # Busca semântica e construção do prompt
│   └── chat.py                 # Interface CLI
└── document.pdf                # PDF para ingestão
```

## Tecnologias utilizadas

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.13 |
| Framework RAG | LangChain |
| Banco de dados | PostgreSQL + pgVector |
| Embeddings | OpenAI `text-embedding-3-small` / Google `models/gemini-embedding-001` |
| LLM | OpenAI `gpt-5-nano` / Google `gemini-2.5-flash-lite` |
| Infraestrutura | Docker & Docker Compose |
