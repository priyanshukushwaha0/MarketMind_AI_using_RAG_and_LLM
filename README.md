## MarketMind_AI_using_RAG_and_LLM

 This is RAG-Powered Stock Market Intelligence chatbot that helps users understand the stock market, investment concepts, analyze companies,stock market books,company 
 fundamentals, and retrieve insights from financial documents using Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Python, LangChain, Hugging-Face,
 FAISS and Streamlit.

---

## Features

- Real-time AI-generated answers using `text-generation`
- Answer stock market and investing questions
- Chat with company fundamentals and investment concepts
- Compare companies (e.g., TCS vs Infosys)
- Explain financial ratios (P/E, ROE, EPS, etc.)
- Retrieve information from stock market books
- Generate context-aware responses using LLMs


---

## Technologies Used

- Python
- LangChain
- RAG (Retrieval-Augmented Generation)
- Large Language Models (LLMs)
- NLP
- Streamlit
- Hugging Face Transformers(Meta-Llama-3-8B-Instruct)
- FAISS Vector Database
- Sentence Transformers

---

## How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MarketMind_AI
   cd MarketMind_AI
2. Install dependencies --> pip install -r requirements.txt

   Run the Streamlit app --> python -m streamlit run market_mind.py


3. File Structure

 ├── market_mind.py                # Response Generation or Main chatbot logic
 ├── connect_memory_with_llm.py    # Information Retrieval
 ├── create_memory_for_llm.py      # Document Indexing
 ├── requirements.txt              # Python dependencies
 ├── README.md                     # Project description
 ├── chatbot_qna.md                # Sample Q&A examples
