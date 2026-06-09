import os
from statistics import mode
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id,HF_TOKEN ):
    # Initialize the base endpoint
    base_llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    chat_llm = ChatHuggingFace(llm=base_llm)
    return chat_llm


def main():
    st.title("Ask Assistant!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Enter your question")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

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
        - Do not speculate about future stock prices or market movements unless explicitly supported by the context.
        - Be concise, factual, and professional.
        - STRICTLY limit your final answer to 5 to 6 lines maximum.

        Context: {context}
        Question: {question}

        Answer:
             """
        
        HUGGINGFACE_REPO_ID="meta-llama/Meta-Llama-3-8B-Instruct" # UNPAID
        HF_TOKEN=os.environ.get("HF_TOKEN") 
        
         
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=False, # True if you want source documents, False if you only want the clean answer
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
             # Invoke the chain
            response=qa_chain.invoke({'query':prompt})
            
             # Extract only the result (no source_documents key will exist now)
            
            result_to_show = response["result"]
            
            # Extract only the result (with source_documents key will exist now)
                #result=response["result"]
                #source_documents=response["source_documents"]
                #result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            
            
            #response="Hi, I am MarketMind AI AI!"
            
            # Display the clean response in the chat
            st.chat_message('Assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'Assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()