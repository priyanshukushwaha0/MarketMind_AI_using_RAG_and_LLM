import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Setup LLM (META LLAMA 3  with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_llm(huggingface_repo_id):
    # Initialize the base endpoint
    base_llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    # Wrap the endpoint to route through the chat completions API
    chat_llm = ChatHuggingFace(llm=base_llm)
    return chat_llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
You are an expert Stock Market and Financial Assistant.

Your task is to answer the user's question using ONLY the information provided in the context.

Instructions:
- Use only the provided context.
- Do not add information from your own knowledge.
- If the answer cannot be found in the context, reply:
  "I don't know based on the available documents."
- Explain financial concepts in simple language.
- Present numerical data, ratios, and statistics accurately.
- Use bullet points when appropriate.
- Do not speculate about future stock prices or market movements unless explicitly supported by the context.
- Be concise, factual, and professional.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query = input("Enter Your Question: ")
response = qa_chain.invoke({'query': user_query})

# Print the results
print("RESULT: ", response["result"]) 
print("SOURCE DOCUMENTS: ", response["source_documents"])