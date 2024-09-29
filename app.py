import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LlamaAIChain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def analyze_website(self, url, question):
        loader = WebBaseLoader([url])
        data = clean_text(loader.load()[0].page_content)

        prompt = PromptTemplate(
            input_variables=["website_content", "question"],
            template="""
            Analyze the following website content and answer the user's question:

            Website Content:
            {website_content}

            User's Question:
            {question}

            Provide a detailed and informative answer based on the website content:
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(website_content=data, question=question)
        return response

def set_page_config():
    st.set_page_config(page_title="Web Analyzer", page_icon="✨", layout="wide")
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .main-title {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .response-area {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stTextInput>div>div>textarea {
        min-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

def website_analysis_interface():
    url = st.text_input("Enter website URL:")
    website_question = st.text_area("Enter your question about the website:", height=100, key="website_question_input")
    if st.button("Analyze"):
        if url and website_question:
            with st.spinner("Analyzing website..."):
                analysis = st.session_state.llama_chain.analyze_website(url, website_question)
                st.markdown("<div class='response-area'>", unsafe_allow_html=True)
                st.write("Analysis:", analysis)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter both a URL and a question.")

def create_streamlit_app():
    set_page_config()

    if 'llama_chain' not in st.session_state:
        st.session_state.llama_chain = LlamaAIChain()

    st.markdown("<h1 class='main-title'>✨ Web Analyzer by Ai Craft Alchemy</h1>", unsafe_allow_html=True)

    # Main content area
    st.markdown("<h2 class='section-title'>Analyze Website</h2>", unsafe_allow_html=True)
    website_analysis_interface()

    # Add footer
    st.markdown("""
    <div class='footer'>
    Developed  by  <a href='https://aicraftalchemy.github.io'>Ai Craft Alchemy</a><br>
    Connect with us: <a href='tel:+917661081043'>+91 7661081043</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_app()
