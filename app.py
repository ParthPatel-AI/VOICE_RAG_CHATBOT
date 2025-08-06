import streamlit as st
import os
import logging
import time
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    logger.warning("Speech recognition not available")

try:
    from utils.rag_utils import load_vectorstore, get_relevant_context
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG utils not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available")

# Safely get API key
def get_api_key():
    """Safely retrieve API key from secrets or environment"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        logger.info("✅ Loaded Gemini API key from st.secrets")
        return api_key
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            logger.info("✅ Loaded Gemini API key from environment variable")
            return api_key
        else:
            logger.error("❌ GEMINI_API_KEY not found in st.secrets or env")
            return None

# Custom CSS - Optimized for deployment
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
    
    .warning-message {
        background: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
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
    
    /* Clean typography */
    h1, h2, h3 {
        color: #1e293b;
    }
    
    p {
        color: #475569;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize app with error handling
@st.cache_resource
def initialize_app():
    """Initialize Parth's AI interview assistant with error handling"""
    try:
        api_key = get_api_key()
        if not api_key:
            return None, None
        
        if GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            logger.info("✅ Gemini API configured")
        else:
            st.error("❌ Google Generative AI not available")
            return None, None
        
        if RAG_AVAILABLE:
            vectorstore = load_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load knowledge base")
                return None, None
        else:
            st.warning("⚠️ RAG functionality not available")
            vectorstore = None
        
        if SPEECH_AVAILABLE:
            recognizer = sr.Recognizer()
            logger.info("✅ Speech recognition initialized")
        else:
            recognizer = None
            logger.info("ℹ️ Speech recognition not available")
        
        logger.info("✅ Application initialized successfully")
        return vectorstore, recognizer
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        logger.error(f"Initialization error: {e}")
        return None, None

# Voice transcription with better error handling
def transcribe_audio_enhanced(recognizer):
    """Voice transcription with deployment-safe error handling"""
    if not SPEECH_AVAILABLE or recognizer is None:
        st.warning("🚫 Voice input not available in this deployment environment")
        return None, False
    
    try:
        with sr.Microphone() as source:
            listening_placeholder = st.empty()
            listening_placeholder.markdown(
                '<div class="listening-indicator">🎤 Listening... Speak your question clearly</div>', 
                unsafe_allow_html=True
            )
            
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
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
        st.warning(f"🚫 Voice input unavailable: {str(e)}")
        return None, False

# Fallback AI response for when RAG is not available
def get_fallback_response(question):
    """Fallback responses when RAG is not available"""
    question_lower = question.lower()
    
    fallback_responses = {
        'experience': "I have 1+ years of experience with internships at L&T (CRNN OCR project with 32% improvement) and AlgoBrain AI (Face recognition with 94% accuracy). I'm currently pursuing B.Tech in AI & Data Science.",
        'project': "My key projects include: OCR system with CRNN (32% improvement), Face Recognition with EfficientNet (94% accuracy), Neurodiagnostic CNN model (90%+ accuracy), and various AI/ML applications using Python, PyTorch, and TensorFlow.",
        'skill': "My technical skills span Python, PyTorch, TensorFlow, OpenCV, LangGraph, FastAPI, Docker, ChromaDB, and various AI/ML frameworks. I specialize in computer vision, NLP, and RAG systems.",
        'background': "I'm Parth Patel, an AI/ML engineer with B.Tech in AI & Data Science. I have hands-on experience from L&T and AlgoBrain AI internships, focusing on computer vision, OCR, and machine learning applications.",
        '100x': "I'm interested in 100x because of their focus on AI agents, automation, and remote-first innovation culture. I believe my rapid learning ability and AI/ML expertise align well with their mission."
    }
    
    for key, response in fallback_responses.items():
        if key in question_lower:
            return response
    
    return "Thank you for your question! While I'd love to provide detailed information about my background, the full knowledge base isn't available in this deployment. Please feel free to ask about my experience, projects, or technical skills!"

# Enhanced AI response with fallback
def ask_gemini_enhanced(question, vectorstore):
    """Enhanced Gemini interaction with fallback handling"""
    try:
        if not GEMINI_AVAILABLE:
            return get_fallback_response(question), False
        
        # Get context if RAG is available
        if RAG_AVAILABLE and vectorstore:
            context = get_relevant_context(vectorstore, question, max_context_length=4000)
        else:
            context = "Limited context available due to deployment constraints."
        
        # Determine response focus
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
        
        prompt = f"""
You are Parth Patel answering an interview question. Use the available information to provide helpful responses.

RESPONSE STRATEGY: {response_focus}

BACKGROUND CONTEXT:
{context}

QUESTION: {question}

RESPONSE GUIDELINES:
- Be conversational and professional
- Use specific examples when possible
- Keep responses 3-5 sentences for most questions
- If specific information isn't available, provide general insights about your background

Answer as Parth Patel:
"""
        
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip(), True
        else:
            return get_fallback_response(question), False
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return get_fallback_response(question), False

# Message display function
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

# Timestamp function
def get_timestamp():
    """Get formatted timestamp in IST"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        ist_time = datetime.now(ist)
        return ist_time.strftime("%I:%M %p")
    except:
        return datetime.now().strftime("%I:%M %p")

# Session metrics display
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

# Main application
def main():
    # Page configuration
    st.set_page_config(
        page_title="🎙️ Parth Patel - AI Interview Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load styling
    load_custom_css()
    
    # Initialize app
    vectorstore, recognizer = initialize_app()
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False
    
    # Header with system status
    voice_status = "✅ Voice Enabled" if SPEECH_AVAILABLE and recognizer else "❌ Voice Disabled"
    rag_status = "✅ RAG Enabled" if RAG_AVAILABLE and vectorstore else "⚠️ Limited Mode"
    
    st.markdown(f"""
    <div class="header-container">
        <h1>🎙️ Interview with Parth Patel</h1>
        <div style="margin: 1rem 0;">
            <span class="tech-badge">AI/ML Engineer</span>
            <span class="tech-badge">100x Candidate</span>
            <span class="tech-badge">{rag_status}</span>
            <span class="tech-badge">{voice_status}</span>
        </div>
        <p>
            <strong>AI-Powered Interview Assistant</strong><br>
            Ask me about my experience, projects, and technical expertise
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status warnings
    if not RAG_AVAILABLE or not vectorstore:
        st.markdown('''
        <div class="warning-message">
            ⚠️ <strong>Limited Mode:</strong> RAG knowledge base not available. Using fallback responses based on core information.
        </div>
        ''', unsafe_allow_html=True)
    
    if not SPEECH_AVAILABLE or not recognizer:
        st.markdown('''
        <div class="warning-message">
            🚫 <strong>Text Mode:</strong> Voice input not available in this deployment environment. Please use text input.
        </div>
        ''', unsafe_allow_html=True)
    
    # Main chat interface
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Welcome message
        if not st.session_state.session_started:
            st.markdown('''
            <div class="welcome-section">
                <h3>👋 Hello! I'm Parth Patel</h3>
                <p>Welcome to my AI-powered interview assistant! This system can answer questions about my background, experience, and projects.</p>
                
                <h4>🎯 Great questions to ask:</h4>
                <ul>
                    <li><strong>Technical:</strong> "Tell me about your OCR project" or "How did you achieve 94% accuracy?"</li>
                    <li><strong>Experience:</strong> "What did you do at L&T?" or "Describe your machine learning background"</li>
                    <li><strong>Projects:</strong> "Walk me through your face recognition system" or "What's your RAG chatbot about?"</li>
                    <li><strong>Career:</strong> "Why 100x?" or "What are your growth areas?"</li>
                </ul>
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
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            voice_disabled = not SPEECH_AVAILABLE or not recognizer
            if st.button("🎤 Voice Question", use_container_width=True, type="primary", disabled=voice_disabled):
                if not voice_disabled:
                    question, success = transcribe_audio_enhanced(recognizer)
                    
                    if success and question:
                        st.markdown(f'''
                        <div class="success-message">
                            🗣️ <strong>Question:</strong> "{question}"
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        with st.spinner("AI assistant is responding..."):
                            answer, success = ask_gemini_enhanced(question, vectorstore)
                        
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
                with st.spinner("AI assistant is responding..."):
                    answer, success = ask_gemini_enhanced(user_input, vectorstore)
                
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
                    transcript = f"""INTERVIEW WITH PARTH PATEL
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Questions: {len(st.session_state.conversation_history)}

{'='*50}

"""
                    for i, chat in enumerate(st.session_state.conversation_history, 1):
                        transcript += f"""Q{i}: {chat['question']}

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
                st.rerun()
        
        # Display metrics if requested
        if st.session_state.get('show_stats', False) and st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 📊 Interview Statistics")
            display_session_metrics()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
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
        st.markdown("### 🔧 System Status")
        
        status_items = []
        if GEMINI_AVAILABLE:
            status_items.append("✅ Gemini AI")
        else:
            status_items.append("❌ Gemini AI")
            
        if RAG_AVAILABLE and vectorstore:
            status_items.append("✅ RAG System")
        else:
            status_items.append("⚠️ Limited Mode")
            
        if SPEECH_AVAILABLE and recognizer:
            status_items.append("✅ Voice Input")
        else:
            status_items.append("❌ Text Only")
        
        for item in status_items:
            st.markdown(f"- {item}")
        
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 📈 Session")
            st.write(f"Questions: {len(st.session_state.conversation_history)}")
            if st.session_state.conversation_history:
                last_time = st.session_state.conversation_history[-1]['timestamp']
                st.write(f"Last: {last_time}")

if __name__ == "__main__":
    main()
