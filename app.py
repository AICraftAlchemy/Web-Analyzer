import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import re
import base64
from gtts import gTTS
import io

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def analyze_website(self, cleaned_text, user_requirements, output_format):
        prompt_analyze = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            
            ### USER REQUIREMENTS:
            {user_requirements}
            
            ### OUTPUT FORMAT:
            {output_format}
            
            ### INSTRUCTION:
            Analyze the scraped text from the website based on the user's requirements.
            Provide a detailed analysis addressing each of the user's requirements.
            Format your response in JSON with the following structure:
            {{
                "summary": "A brief summary of the website content",
                "analysis": [
                    {{
                        "requirement": "User's requirement",
                        "response": {{"format": "The format of the response (points or paragraph)",
                                      "content": "The actual content of the response"}}
                    }},
                    ...
                ]
            }}
            If the output format is "points", provide the content as a list of concise, informative points.
            If the output format is "paragraph", provide the content as a well-structured paragraph.
            Only return the valid JSON.
            
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_analyze = prompt_analyze | self.llm
        res = chain_analyze.invoke(input={"page_data": cleaned_text, "user_requirements": user_requirements, "output_format": output_format})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse analysis results.")
        return res

    def generate_headlines(self, topic, num_headlines):
        prompt_headlines = PromptTemplate.from_template(
            """
            Generate {num_headlines} engaging and creative headlines for the following topic:
            
            TOPIC: {topic}
            
            INSTRUCTIONS:
            1. Each headline should be unique and captivating.
            2. Use a variety of headline styles (e.g., question, how-to, listicle, etc.).
            3. Aim for a mix of emotional appeal and information.
            4. Keep each headline under 80 characters.
            5. Format the output as a JSON array of strings.
            
            OUTPUT JSON:
            """
        )
        chain_headlines = prompt_headlines | self.llm | JsonOutputParser()
        res = chain_headlines.invoke({"topic": topic, "num_headlines": num_headlines})
        return res

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_audio_player(audio_bytes):
    b64 = base64.b64encode(audio_bytes.getvalue()).decode()
    return f'<audio autoplay="true" controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

def set_theme(is_dark_mode):
    if is_dark_mode:
        st.markdown("""
        <style>
        :root {
            --background-color: #1E1E1E;
            --text-color: #E0E0E0;
            --card-background: #2C2C2C;
            --button-color: #007BFF;
            --button-hover: #0056b3;
            --accent-color: #FF4081;
            --secondary-color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --background-color: #F0F2F6;
            --text-color: #333333;
            --card-background: #FFFFFF;
            --button-color: #4CAF50;
            --button-hover: #45a049;
            --accent-color: #FF4081;
            --secondary-color: #007BFF;
        }
        </style>
        """, unsafe_allow_html=True)

def create_streamlit_app(llm, clean_text):
    st.set_page_config(page_title="Enhanced Web Analyzer & Headline Generator", page_icon="🌐", layout="wide")
    
    # Initialize session state for theme
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True

    # Apply theme
    set_theme(st.session_state.dark_mode)

    # Custom CSS for the app
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }
        .stApp {
            background-color: var(--background-color);
        }
        .main {
            background-color: var(--background-color);
        }
        .title {
            text-align: center;
            color: var(--text-color);
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(45deg, var(--button-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: var(--text-color);
            font-size: 1.5em;
            margin-bottom: 30px;
            font-weight: 300;
        }
        .container {
            padding: 30px;
            max-width: 1000px;
            margin: 0 auto;
            background-color: var(--card-background);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .input-box {
            margin-bottom: 25px;
        }
        .input-box input, .input-box textarea {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: var(--background-color);
            color: var(--text-color);
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .input-box input:focus, .input-box textarea:focus {
            outline: none;
            box-shadow: 0 0 0 2px var(--accent-color);
        }
        .stButton > button {
            background: linear-gradient(45deg, var(--button-color), var(--accent-color));
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 700;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .result-card {
            background-color: var(--card-background);
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }
        .result-card h3 {
            color: var(--accent-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 700;
        }
        .footer {
            text-align: center;
            color: var(--text-color);
            font-size: 1em;
            margin-top: 40px;
            padding: 20px;
            background-color: var(--card-background);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .footer:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .footer a {
            color: var(--accent-color);
            text-decoration: none;
            transition: all 0.3s ease;
            font-weight: 700;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .toggle-container {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 10px;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background-color: var(--accent-color);
        }
        input:checked + .toggle-slider:before {
            transform: translateX(26px);
        }
        .toggle-label {
            margin-right: 10px;
            color: var(--text-color);
            font-weight: 700;
        }
        .headline-list {
            list-style-type: none;
            padding-left: 0;
        }
        .headline-item {
            background-color: var(--background-color);
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
        .headline-item:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .headline-number {
            background-color: var(--accent-color);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-right: 15px;
            font-weight: bold;
        }
        .tab-content {
            padding: 20px;
            background-color: var(--card-background);
            border-radius: 0 0 15px 15px;
        }
        .stSlider > div > div > div {
            background-color: var(--accent-color);
        }
        .stSlider > div > div > div > div {
            color: var(--text-color);
        }
        .analysis-point {
            background-color: var(--background-color);
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .analysis-point:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }
        .analysis-paragraph {
            background-color: var(--background-color);
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            line-height: 1.6;
        }
        .analysis-paragraph:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Theme toggle
    col1, col2 = st.columns([4, 1])
    with col2:
        st.markdown("""
        <div class="toggle-container">
            <span class="toggle-label">Dark Mode</span>
            <label class="toggle-switch">
                <input type="checkbox" id="theme-toggle">
                <span class="toggle-slider"></span>
            </label>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Toggle Theme", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.experimental_rerun()

    st.markdown("<div class='title'>🌐 Enhanced Web Analyzer & Headline Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Uncover insights, create engaging headlines, and listen to analysis with AI</div>", unsafe_allow_html=True)

    st.markdown("<div class='container'>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Website Analyzer",  "Headline Generator"])
    
    with tab1:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        url_input = st.text_input("🔗 Enter Website URL:", value="", placeholder="https://example.com")
        user_requirements = st.text_area(
            "🔍 What insights are you looking for?", 
            value="",
            placeholder="E.g., Analyze main topics, identify target audience, list products/services, evaluate user experience...",
            height=150
        )
        output_format = st.radio("Select output format:", ("Points", "Paragraphs"))

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            analyze_button = st.button("🚀 Analyze Website", key='analyze_button', help="Click to start the in-depth website analysis")

        if analyze_button and url_input and user_requirements:
            try:
                with st.spinner("🕵️‍♀️ Our AI is diving deep into the website... Please wait."):
                    loader = WebBaseLoader([url_input])
                    data = clean_text(loader.load().pop().page_content)
                    analysis_result = llm.analyze_website(data, user_requirements, output_format.lower())

                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>🎯 Analysis Results</h3>", unsafe_allow_html=True)
                    st.markdown(f"**📌 Executive Summary:**\n\n{analysis_result['summary']}")
                    
                    full_analysis_text = f"Executive Summary: {analysis_result['summary']}\n\n"
                    
                    for item in analysis_result['analysis']:
                        st.markdown(f"**🔍 {item['requirement']}**")
                        if item['response']['format'] == 'points':
                            for point in item['response']['content']:
                                st.markdown(f"<div class='analysis-point'>• {point}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='analysis-paragraph'>{item['response']['content']}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        full_analysis_text += f"{item['requirement']}:\n"
                        if item['response']['format'] == 'points':
                            full_analysis_text += "\n".join([f"• {point}" for point in item['response']['content']])
                        else:
                            full_analysis_text += item['response']['content']
                        full_analysis_text += "\n\n"
                    
                    audio_bytes = text_to_speech(full_analysis_text)
                    st.markdown("### 🔊 Listen to the Analysis")
                    st.markdown(get_audio_player(audio_bytes), unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"😕 Oops! We encountered an issue: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        headline_topic = st.text_input("📝 Enter the topic for headlines:", placeholder="E.g., Artificial Intelligence in Healthcare")
        num_headlines = st.slider("🔢 Number of headlines to generate:", min_value=1, max_value=10, value=5)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            generate_button = st.button("✨ Generate Headlines", key='generate_button')
        
        if generate_button and headline_topic:
            try:
                with st.spinner("🧠 Crafting engaging headlines... Just a moment!"):
                    headlines = llm.generate_headlines(headline_topic, num_headlines)
                    
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>🎭 Generated Headlines</h3>", unsafe_allow_html=True)
                    st.markdown("<ul class='headline-list'>", unsafe_allow_html=True)
                    for i, headline in enumerate(headlines, 1):
                        st.markdown(f"""
                        <li class='headline-item'>
                            <span class='headline-number'>{i}</span>
                            {headline}
                        </li>
                        """, unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"😕 Oops! We couldn't generate headlines: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Need expert assistance? Contact our support team at <a href='mailto:aicraftalchemy@gmail.com'>aicraftalchemy@gmail.com</a> | 📞 7661081043</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    chain = Chain()
    create_streamlit_app(chain, clean_text)