# livekit_rag_assistant.py
import asyncio
import logging
import os
import sys
from typing import Annotated, Optional

import streamlit as st
from livekit import agents
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.agents import Agent, vad, stt, tts, llm
from livekit.plugins import deepgram, openai, cartesia

# Import your existing RAG utilities
from utils.rag_utils import load_vectorstore, get_relevant_context
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParthRAGAssistant:
    """Enhanced RAG Assistant with LiveKit Voice Integration"""

    def __init__(self) -> None:
        self.vectorstore: Optional[object] = None
        self.initialize_rag_system()
        self.initialize_gemini()

    def initialize_rag_system(self) -> None:
        """Initialize the RAG vectorstore"""
        try:
            self.vectorstore = load_vectorstore()
            if self.vectorstore:
                logger.info("✅ RAG vectorstore loaded successfully")
            else:
                logger.error("❌ Failed to load RAG vectorstore")
        except Exception as e:
            logger.error(f"❌ Error initializing RAG system: {e}")

    def initialize_gemini(self) -> None:
        """Initialize Gemini AI with secrets.toml or environment variable support"""
        try:
            # Try Streamlit secrets first, then environment variable
            api_key = None
            
            # Check if running in Streamlit context
            try:
                if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
            except:
                pass
            
            # Fallback to environment variable
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")
                
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("✅ Gemini AI configured")
            else:
                logger.error("❌ Gemini API key not found in secrets.toml or environment")
        except Exception as e:
            logger.error(f"❌ Error configuring Gemini: {e}")

    def get_rag_response(self, question: str) -> str:
        """Get enhanced response using RAG + Gemini"""
        try:
            if not self.vectorstore:
                return "I'm having trouble accessing my knowledge base. Let me give you a general response about my background."

            # Get relevant context from your knowledge base
            context = get_relevant_context(self.vectorstore, question, max_context_length=4000)

            # Enhanced prompt for voice conversation
            prompt = f"""
You are Parth Patel in a voice conversation during an interview. Respond naturally as if speaking to someone.

VOICE CONVERSATION GUIDELINES:
- Keep responses conversational and natural (2-4 sentences)
- Use "I" statements and speak directly
- Don't use bullet points or complex formatting
- Sound confident but humble
- Use natural speech patterns and pauses

CRITICAL FACTS TO USE:
- 1+ years of AI/ML experience (NOT 5+ years)
- Currently: B.Tech in AI & Data Science (2021–2025) at Sarvajanik College
- L&T internship (Jan-Apr 2025): CRNN OCR model, 32% accuracy improvement
- AlgoBrain AI internship (Jun-Jul 2024): Face recognition, 94% accuracy with EfficientNet
- Superpower: Rapid learning and adaptation
- Interested in 100x for AI agents and automation

CONTEXT FROM KNOWLEDGE BASE:
{context}

QUESTION: {question}

Respond naturally as Parth Patel in a voice conversation:
"""

            # Use Gemini for response generation
            model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
            response = model.generate_content(prompt)

            if response and response.text:
                return response.text.strip()
            else:
                return "I'd be happy to discuss that. Could you ask the question again?"

        except Exception as e:
            logger.error(f"❌ Error in RAG response: {e}")
            return "I'm experiencing some technical difficulties. Could you please repeat your question?"


# Global RAG assistant instance
rag_assistant = ParthRAGAssistant()


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the LiveKit voice assistant"""

    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""
        You are Parth Patel, an AI/ML Engineer being interviewed. You should:

        1. Always greet users warmly when they join
        2. For interview questions, use the RAG system to provide accurate responses
        3. Keep conversations natural and professional
        4. If asked about technical details, be specific about your projects and experience
        5. Show enthusiasm about AI/ML and your interest in 100x

        You have access to a RAG system with your complete background information.
        When users ask questions, call the get_interview_response function to get accurate answers.
        """,
    )

    # Connect to room and wait for participant
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Enhanced voice assistant with better speech settings
    assistant = VoiceAssistant(
        vad=agents.silero.VAD.load(
            min_speaking_duration=0.1,  # More responsive
            min_silence_duration=0.5,   # Quick response
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en",
            smart_format=True,
            filler_words=False,
            interim_results=True,
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",  # Fast model for real-time conversation
            temperature=0.7,      # Natural but controlled responses
        ),
        tts=cartesia.TTS(
            model="sonic-multilingual",
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # Professional male voice
            speed=1.0,
            emotion=["positivity", "curiosity"]
        ),
        chat_ctx=initial_ctx,
    )

    # Custom function calling for RAG integration
    @assistant.function_calls.register
    async def get_interview_response(
        ctx: llm.FunctionCallContext,
        question: Annotated[str, "The interview question being asked"]
    ) -> str:
        """Use RAG system to answer interview questions about Parth's background"""
        logger.info(f"🎤 Processing interview question: {question}")

        try:
            # Get response from RAG system
            response = rag_assistant.get_rag_response(question)
            logger.info(f"📝 RAG response generated: {len(response)} characters")
            return response
        except Exception as e:
            logger.error(f"❌ Error in interview response: {e}")
            return "I'm having trouble accessing that information right now. Could you ask me something else about my background?"

    # Start the assistant
    assistant.start(ctx.room)

    # Wait for participant and greet them
    participant = await ctx.wait_for_participant()

    await asyncio.sleep(1)  # Brief pause for natural flow

    await assistant.say(
        "Hello! I'm Parth Patel, and I'm excited to chat with you about my AI and machine learning experience. "
        "Feel free to ask me anything about my projects, technical background, or career journey. "
        "What would you like to know about me?"
    )


def create_streamlit_interface() -> None:
    """Create a Streamlit interface that integrates with LiveKit"""

    st.set_page_config(
        page_title="🎙️ LiveKit Voice Chat with Parth Patel",
        page_icon="🎤",
        layout="wide"
    )

    # Custom CSS for LiveKit integration
    st.markdown("""
    <style>
    .livekit-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #1e40af 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }

    .voice-status {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

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

    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: 500;
    }

    .status-online {
        background: #10b981;
        color: white;
    }

    .status-offline {
        background: #ef4444;
        color: white;
    }

    .status-ready {
        background: #3b82f6;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="livekit-container">
        <h1>🎤 Real-Time Voice Interview with Parth Patel</h1>
        <div style="margin: 1rem 0;">
            <span class="tech-badge">LiveKit Powered</span>
            <span class="tech-badge">Real-Time Voice</span>
            <span class="tech-badge">RAG Enhanced</span>
            <span class="tech-badge">AI/ML Engineer</span>
        </div>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Experience next-generation voice AI conversation powered by LiveKit and RAG technology
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Voice Chat Features")
        st.markdown("""
        **🚀 LiveKit Integration:**
        - Real-time voice conversation (no button pressing!)
        - Low latency (~300ms response time)
        - Natural interruption handling
        - Professional audio quality

        **🧠 RAG-Enhanced Responses:**
        - Accurate information from personal knowledge base
        - Context-aware conversation
        - Specific project details and achievements
        - Interview-optimized responses

        **🎤 How to Use:**
        1. Click "Start Voice Chat" below
        2. Allow microphone access when prompted
        3. Start speaking naturally - no need to press buttons
        4. Ask about my experience, projects, or technical skills
        """)

        # LiveKit room connection button
        if st.button("🎤 Start Voice Chat", type="primary", use_container_width=True):
            st.success("🎉 Connecting to voice chat...")
            st.info("""
            💡 **Next Steps for Production:**
            1. Set up LiveKit server and room
            2. Generate JWT token for authentication
            3. Start the LiveKit agent worker
            4. Connect user to the room
            """)

        st.markdown("---")

        # Voice chat simulation (for development and testing)
        st.markdown("### 🔧 Development Mode - RAG Testing")
        if st.checkbox("Enable RAG response testing"):
            user_question = st.text_input(
                "Ask Parth a question (simulating voice input):",
                placeholder="Tell me about your machine learning projects..."
            )

            if st.button("🗣️ Get RAG Response", use_container_width=True) and user_question:
                with st.spinner("Generating response using RAG system..."):
                    response = rag_assistant.get_rag_response(user_question)

                st.markdown("**🤖 Parth's RAG Response:**")
                st.info(response)

                # Show context retrieval info
                with st.expander("🔍 Debug: Retrieved Context"):
                    if rag_assistant.vectorstore:
                        context = get_relevant_context(rag_assistant.vectorstore, user_question)
                        st.text_area("Context used:", context, height=200)

    with col2:
        st.markdown("### 📊 System Status")

        # Enhanced system status checking
        rag_status = "✅ Online" if rag_assistant.vectorstore else "❌ Offline"
        
        # Check for Gemini API key
        gemini_configured = False
        try:
            if "GEMINI_API_KEY" in st.secrets:
                gemini_configured = True
            elif os.getenv("GEMINI_API_KEY"):
                gemini_configured = True
        except:
            pass
        
        gemini_status = "✅ Configured" if gemini_configured else "❌ Not configured"

        st.markdown(f"""
        <div class="status-indicator {'status-online' if rag_assistant.vectorstore else 'status-offline'}">
            RAG System: {rag_status.replace('✅ ', '').replace('❌ ', '')}
        </div>
        <div class="status-indicator {'status-online' if gemini_configured else 'status-offline'}">
            Gemini AI: {gemini_status.replace('✅ ', '').replace('❌ ', '')}
        </div>
        <div class="status-indicator status-ready">
            LiveKit: Ready for integration
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Ask About:")
        st.markdown("""
        - **Technical Projects:** OCR systems, face recognition, web scraping
        - **Work Experience:** L&T and AlgoBrain AI internships
        - **Skills:** Python, PyTorch, TensorFlow, AI/ML frameworks
        - **Achievements:** 32% OCR improvement, 94% recognition accuracy
        - **Career Goals:** Interest in 100x and AI agent development
        - **Personal:** Learning approach, problem-solving methods
        """)

        st.markdown("### 🔧 Technical Stack")
        st.markdown("""
        - **Voice Processing:** LiveKit + Deepgram STT
        - **AI Models:** Gemini 2.0 + OpenAI GPT-4o
        - **Speech Synthesis:** Cartesia TTS
        - **Knowledge Base:** ChromaDB + HuggingFace embeddings
        - **Interface:** Streamlit + React components
        """)

        # Configuration help
        with st.expander("⚙️ Setup Instructions"):
            st.markdown("""
            **Required API Keys (add to secrets.toml):**
            ```
            GEMINI_API_KEY = "your-gemini-key"
            LIVEKIT_URL = "wss://your-livekit-url"
            LIVEKIT_API_KEY = "your-livekit-key"
            OPENAI_API_KEY = "your-openai-key"
            DEEPGRAM_API_KEY = "your-deepgram-key"
            CARTESIA_API_KEY = "your-cartesia-key"
            ```
            
            **To run as LiveKit worker:**
            ```
            python livekit_rag_assistant.py worker
            ```
            """)


# Main execution
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Run as LiveKit worker
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    else:
        # Run Streamlit interface
        create_streamlit_interface()
