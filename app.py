import streamlit as st
import speech_recognition as sr
from utils.rag_utils import load_vectorstore, get_relevant_context
import google.generativeai as genai
import time
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🎨 Simple & Classy Tech Color Palette CSS
def load_custom_css():
    st.markdown("""
    <style>
    /* Import clean professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Clean global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Simple gradient background - Tech Blue/Purple */
    .main {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #1e40af 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean header container */
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .header-container h1 {
        color: #1e40af;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .header-container p {
        color: #475569;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Simple chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Clean message bubbles */
    .user-message {
        background: #1e40af;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);
        font-weight: 500;
    }
    
    .bot-message {
        background: #f8fafc;
        color: #334155;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        max-width: 80%;
        border-left: 4px solid #06b6d4;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        line-height: 1.6;
    }
    
    /* Simple status indicators */
    .listening-indicator {
        background: #fef3c7;
        color: #f59e0b;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        border: 2px solid #fbbf24;
    }
    
    /* Clean input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1e40af;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
    }
    
    /* Professional button styling */
    .stButton > button {
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        height: 3rem;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af, #3730a3);
        color: white;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Clean metrics */
    .metric-container {
        background: linear-gradient(135deg, #1e40af, #3730a3);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.25);
    }
    
    .metric-container h3 {
        font-size: 2rem;
        margin: 0;
    }
    
    .metric-container p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .metric-container h2 {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
    }
    
    /* Professional welcome section */
    .welcome-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 4px solid #06b6d4;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .welcome-section h3 {
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    .welcome-section h4 {
        color: #475569;
        margin: 1.5rem 0 0.5rem 0;
    }
    
    /* Success/Error messages */
    .success-message {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .error-message {
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    /* Professional badges */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    /* Clean sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Section dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Professional spacing */
    .section-spacing {
        margin: 2rem 0;
    }
    
    /* Clean typography */
    h1, h2, h3 {
        color: #1e293b;
    }
    
    p {
        color: #475569;
        line-height: 1.6;
    }
    
    /* Remove excessive animations - keep it professional */
    * {
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# 🔐 Initialize app (keeping the same function)
@st.cache_resource
def initialize_app():
    """Initialize Parth's AI interview assistant"""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            logger.info("✅ Gemini API configured")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("✅ Gemini API configured from environment")
            else:
                st.error("❌ Gemini API key not found")
                return None, None
        
        vectorstore = load_vectorstore()
        if vectorstore is None:
            st.error("❌ Failed to load knowledge base")
            return None, None
        
        recognizer = sr.Recognizer()
        logger.info("✅ Application initialized successfully")
        return vectorstore, recognizer
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None, None

# 🎤 Voice transcription (simplified)
def transcribe_audio_enhanced(recognizer):
    """Clean voice transcription"""
    try:
        with sr.Microphone() as source:
            listening_placeholder = st.empty()
            listening_placeholder.markdown(
                '<div class="listening-indicator">🎤 Listening... Speak your question clearly</div>', 
                unsafe_allow_html=True
            )
            
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            listening_placeholder.empty()
            
            with st.spinner("Processing your voice..."):
                text = recognizer.recognize_google(audio)
                
            return text, True
            
    except sr.WaitTimeoutError:
        st.warning("⏱️ No speech detected. Please try again.")
        return None, False
    except sr.UnknownValueError:
        st.warning("🤔 Could not understand. Please speak more clearly.")
        return None, False
    except Exception as e:
        st.error(f"🚫 Microphone error: {str(e)}")
        return None, False

# 🧠 AI response (keeping the same enhanced function)
def ask_gemini_enhanced(question, vectorstore):
    """Enhanced Gemini interaction with comprehensive keyword awareness"""
    try:
        # Get relevant context with enhanced keyword matching
        context = get_relevant_context(vectorstore, question, max_context_length=4000)
        
        # Determine response focus based on question keywords
        question_lower = question.lower()
        response_focus = ""
        
        if any(term in question_lower for term in ['project', 'built', 'developed', 'system']):
            response_focus = "Focus on providing comprehensive details about relevant projects with technologies, achievements, and metrics."
        elif any(term in question_lower for term in ['experience', 'work', 'background', 'intern']):
            response_focus = "Focus on work experience at L&T and AlgoBrain AI with specific achievements and responsibilities."
        elif any(term in question_lower for term in ['skill', 'technical', 'technology', 'programming']):
            response_focus = "Focus on technical skills across AI/ML, programming languages, frameworks, and tools."
        elif any(term in question_lower for term in ['superpower', 'strength', 'ability']):
            response_focus = "Focus on rapid learning and adaptation abilities with specific examples."
        elif any(term in question_lower for term in ['growth', 'improve', 'future', 'goal']):
            response_focus = "Focus on the three growth areas: leadership, system architecture, and product strategy."
        elif any(term in question_lower for term in ['100x', 'company', 'why']):
            response_focus = "Focus on interest in 100x, AI agents, automation, and remote-first innovation culture."
        
        # Enhanced prompt with keyword awareness
        prompt = f"""
You are Parth Patel answering an interview question. Use ONLY the factual information from the context below.

RESPONSE STRATEGY: {response_focus}

CRITICAL INSTRUCTIONS:
- Use EXACT details from context (numbers, percentages, company names, dates, technologies)
- Your experience is 1+ years (NOT 5+ years)
- L&T internship: Jan 2025 – Apr 2025 (CRNN OCR, 32% improvement, 1000+ docs/day, Power BI)
- AlgoBrain AI internship: Jun 2024 – Jul 2024 (Face recognition, 94% accuracy, EfficientNet, Flask APIs)
- Education: B.Tech in AI & Data Science (2021–2025) at Sarvajanik College, Surat, Gujarat
- 7 Major Projects: Web Scraping (GPT-4), OCR (CRNN), Neurodiagnostic (CNN), Face Recognition (EfficientNet), Sentiment Analysis (BERT), Neural Network (NumPy), RAG Chatbot (Current)
- Key Technologies: Python, PyTorch, TensorFlow, OpenCV, LangGraph, FastAPI, Docker, ChromaDB
- Achievements: 32% OCR improvement, 94% face recognition, 90%+ tumor classification, 60% chatbot response reduction

YOUR ACTUAL BACKGROUND CONTEXT:
{context}

QUESTION: {question}

RESPONSE GUIDELINES:
- 3-5 sentences for most questions
- Be conversational but factually precise
- Use specific examples and metrics from your actual experience
- Don't mix up different projects or achievements
- If information isn't in context, say "That's not detailed in my specific background"

Answer authentically as Parth Patel using ONLY verified information from the context:
"""
        
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip(), True
        else:
            return "I'd be happy to answer that based on my specific background. Could you rephrase the question?", False
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm experiencing technical difficulties. Please try your question again.", False

# 💬 Simple message display
def display_chat_message(message, is_user=False, timestamp=None):
    """Clean message display"""
    timestamp_str = f" • {timestamp}" if timestamp else ""
    
    if is_user:
        st.markdown(f'''
        <div class="user-message">
            <strong>👨‍💼 You{timestamp_str}:</strong><br>
            {message}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="bot-message">
            <strong>🤖 Parth{timestamp_str}:</strong><br>
            {message}
        </div>
        ''', unsafe_allow_html=True)

def get_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%I:%M %p")

# 📊 Clean metrics display
def display_session_metrics():
    """Simple session metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-container">
            <h3>💬</h3>
            <p>Questions</p>
            <h2>{len(st.session_state.conversation_history)}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        technical_count = 0
        if st.session_state.conversation_history:
            technical_count = sum(1 for chat in st.session_state.conversation_history 
                                if any(term in chat['question'].lower() 
                                      for term in ['technical', 'project', 'ai', 'ml', 'python']))
        
        st.markdown(f'''
        <div class="metric-container">
            <h3>🔬</h3>
            <p>Technical</p>
            <h2>{technical_count}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_length = 0
        if st.session_state.conversation_history:
            total_length = sum(len(chat['answer']) for chat in st.session_state.conversation_history)
            avg_length = total_length // len(st.session_state.conversation_history)
        
        st.markdown(f'''
        <div class="metric-container">
            <h3>📝</h3>
            <p>Avg Length</p>
            <h2>{avg_length}</h2>
        </div>
        ''', unsafe_allow_html=True)

# 🏠 Main application with clean design
def main():
    # Page configuration
    st.set_page_config(
        page_title="🎙️ Parth Patel - AI Interview Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load clean styling
    load_custom_css()
    
    # Initialize app
    vectorstore, recognizer = initialize_app()
    
    if vectorstore is None or recognizer is None:
        st.stop()
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False
    
    # Clean header
    st.markdown("""
    <div class="header-container">
        <h1>🎙️ Interview with Parth Patel</h1>
        <div style="margin: 1rem 0;">
            <span class="tech-badge">AI/ML Engineer</span>
            <span class="tech-badge">100x Candidate</span>
            <span class="tech-badge">RAG Expert</span>
            <span class="tech-badge">Voice Enabled</span>
        </div>
        <p>
            <strong>AI-Powered Interview Assistant</strong><br>
            Ask me about my experience, projects, and technical expertise
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chat interface
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Welcome message
        if not st.session_state.session_started:
            st.markdown('''
            <div class="welcome-section">
                <h3>👋 Hello! I'm Parth Patel</h3>
                <p>Welcome to my AI-powered interview assistant! This system is trained on my personal knowledge base and can answer questions about my background, experience, and projects.</p>
                
                <h4>🎯 Great questions to ask:</h4>
                <ul>
                    <li><strong>Technical:</strong> "Tell me about your OCR project" or "How did you achieve 94% accuracy?"</li>
                    <li><strong>Experience:</strong> "What did you do at L&T?" or "Describe your machine learning background"</li>
                    <li><strong>Projects:</strong> "Walk me through your face recognition system" or "What's your RAG chatbot about?"</li>
                    <li><strong>Career:</strong> "Why 100x?" or "What are your growth areas?"</li>
                </ul>
                
                <p><strong>You can use voice or text to ask your questions!</strong></p>
            </div>
            ''', unsafe_allow_html=True)
            st.session_state.session_started = True
        
        # Display conversation history
        for i, chat in enumerate(st.session_state.conversation_history):
            display_chat_message(chat['question'], is_user=True, timestamp=chat['timestamp'])
            display_chat_message(chat['answer'], is_user=False, timestamp=chat['timestamp'])
            
            if i < len(st.session_state.conversation_history) - 1:
                st.markdown("---")
        
        st.markdown("---")
        
        # Input interface
        st.markdown("### 🎯 Ask Your Question")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Voice Question", use_container_width=True, type="primary"):
                question, success = transcribe_audio_enhanced(recognizer)
                
                if success and question:
                    st.markdown(f'''
                    <div class="success-message">
                        🗣️ <strong>Question:</strong> "{question}"
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    with st.spinner("Parth is responding..."):
                        answer, success = ask_gemini_enhanced(question, vectorstore)
                    
                    if success:
                        st.session_state.conversation_history.append({
                            'question': question,
                            'answer': answer,
                            'timestamp': get_timestamp()
                        })
                        st.rerun()
        
        with col2:
            user_input = st.text_input(
                "💬 Type your question:", 
                placeholder="Ask about my experience, projects, or skills...",
                key="user_input"
            )
            
            if st.button("📤 Send Question", use_container_width=True, type="secondary") and user_input:
                with st.spinner("Parth is responding..."):
                    answer, success = ask_gemini_enhanced(user_input, vectorstore)
                
                if success:
                    st.session_state.conversation_history.append({
                        'question': user_input,
                        'answer': answer,
                        'timestamp': get_timestamp()
                    })
                    st.rerun()
        
        # Session controls
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 New Interview", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.session_started = False
                st.rerun()
        
        with col2:
            if st.button("💾 Save Transcript", use_container_width=True):
                if st.session_state.conversation_history:
                    transcript = f"""
INTERVIEW WITH PARTH PATEL
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Questions: {len(st.session_state.conversation_history)}

{'='*50}

"""
                    for i, chat in enumerate(st.session_state.conversation_history, 1):
                        transcript += f"""
Q{i}: {chat['question']}

A{i}: {chat['answer']}

{'-'*30}
"""
                    
                    st.download_button(
                        "📄 Download",
                        transcript,
                        f"parth_interview_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        use_container_width=True
                    )
                else:
                    st.info("Start the interview first!")
        
        with col3:
            if st.button("📊 Show Stats", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)
        
        # Display metrics if requested
        if st.session_state.get('show_stats', False):
            st.markdown("---")
            st.markdown("### 📊 Interview Statistics")
            display_session_metrics()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clean sidebar
    with st.sidebar:
        st.markdown("### 🎯 Interview Guide")
        st.markdown("""
        **Question Categories:**
        
        **🔬 Technical:**
        - OCR and Computer Vision
        - Face Recognition Systems
        - NLP and Sentiment Analysis
        - AI Agents and Automation
        
        **💼 Professional:**
        - L&T and AlgoBrain AI experience
        - Leadership and mentorship
        - Published articles
        - Career achievements
        
        **🚀 Personal:**
        - Learning approach
        - Problem-solving methods
        - Growth areas
        - 100x interest
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        st.markdown("""
        **Features:**
        - RAG-powered responses
        - Voice & text input
        - Personal knowledge base
        - Interview-optimized
        
        **Tech Stack:**
        - Streamlit UI
        - Google Gemini AI
        - ChromaDB vectors
        - Speech recognition
        """)
        
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 📈 Session")
            st.write(f"Questions: {len(st.session_state.conversation_history)}")
            if st.session_state.conversation_history:
                last_time = st.session_state.conversation_history[-1]['timestamp']
                st.write(f"Last: {last_time}")
        else:
            st.markdown("---")
            st.markdown("### 🆕 Ready")
            st.write("Ask your first question!")

if __name__ == "__main__":
    main()
