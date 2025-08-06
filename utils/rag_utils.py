# utils/rag_utils.py
'''
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

def load_vectorstore(file_path="knowledge_base/personal_info.txt", persist_dir="chroma_store"):
    # Load documents from text file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(documents)

    # Use HuggingFace sentence-transformer model for embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create or load Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore

'''

# utils/rag_utils.py
# utils/rag_utils.py
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalRAGSystem:
    """RAG system specifically for Parth's personal_info.txt knowledge base"""
    
    def __init__(self, file_path="knowledge_base/personal_info.txt", persist_dir="chroma_store"):
        self.file_path = file_path
        self.persist_dir = persist_dir
        self.embeddings = None
        self.vectorstore = None
        
    def initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return False
    
    def load_personal_document(self):
        """Load Parth's personal_info.txt document"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            # Check if personal_info.txt exists
            if not os.path.exists(self.file_path):
                st.error(f"❌ {self.file_path} not found! Please ensure your personal_info.txt file exists.")
                return []
            
            # Load the specific file
            loader = TextLoader(self.file_path, encoding="utf-8")
            documents = loader.load()
            
            logger.info(f"✅ Loaded personal_info.txt with {len(documents)} document(s)")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading personal_info.txt: {e}")
            st.error(f"Error loading knowledge base: {e}")
            return []
    
    def process_documents(self, documents):
        """Split personal info into optimized chunks"""
        if not documents:
            logger.warning("⚠️ No documents to process")
            return []
        
        # Optimized text splitter for your personal info structure
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=150,  # Good overlap for context
            length_function=len,
            separators=["\n\n", "\n", "**", "- ", ". ", " ", ""]
        )
        
        doc_chunks = text_splitter.split_documents(documents)
        
        # Add metadata to chunks for better tracking
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata.update({
                "source": "personal_info.txt",
                "chunk_id": i,
                "total_chunks": len(doc_chunks)
            })
        
        logger.info(f"✅ Created {len(doc_chunks)} chunks from personal_info.txt")
        return doc_chunks
    
    def create_vectorstore(self, doc_chunks):
        """Create ChromaDB vectorstore from personal info chunks"""
        try:
            if not self.embeddings:
                if not self.initialize_embeddings():
                    raise Exception("Failed to initialize embeddings")
            
            # Clear existing vectorstore if it exists
            if os.path.exists(self.persist_dir):
                import shutil
                shutil.rmtree(self.persist_dir)
                logger.info("🗑️ Cleared existing vectorstore")
            
            # Create new vectorstore
            if not doc_chunks:
                raise Exception("No document chunks available for vectorstore creation")
            
            self.vectorstore = Chroma.from_documents(
                documents=doc_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="parth_personal_info"
            )
            
            # Persist the vectorstore
            self.vectorstore.persist()
            
            logger.info("✅ Created new ChromaDB vectorstore from personal_info.txt")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            st.error(f"Failed to create knowledge base: {e}")
            return None

@st.cache_resource
def load_vectorstore(file_path="knowledge_base/personal_info.txt", persist_dir="chroma_store"):
    """Load vectorstore from personal_info.txt with caching"""
    try:
        rag_system = PersonalRAGSystem(file_path, persist_dir)
        
        # Load personal document
        documents = rag_system.load_personal_document()
        if not documents:
            return None
        
        # Process documents
        doc_chunks = rag_system.process_documents(documents)
        if not doc_chunks:
            return None
        
        # Create vectorstore
        vectorstore = rag_system.create_vectorstore(doc_chunks)
        
        if vectorstore is None:
            st.error("❌ Failed to create vectorstore from personal_info.txt")
            return None
        
        # Verify vectorstore works
        try:
            test_docs = vectorstore.similarity_search("Parth Patel", k=1)
            if test_docs:
                logger.info("✅ Vectorstore verification successful")
            else:
                logger.warning("⚠️ Vectorstore created but no documents found in test search")
        except Exception as e:
            logger.warning(f"⚠️ Vectorstore verification failed: {e}")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {e}")
        logger.error(f"Vectorstore loading error: {e}")
        return None

def search_knowledge_base(vectorstore, query, k=5):
    """Enhanced search with better keyword preprocessing"""
    try:
        if vectorstore is None:
            logger.error("❌ Vectorstore is None")
            return []
        
        # Enhanced query preprocessing for better matches
        enhanced_query = query
        
        # Expand common abbreviations and synonyms
        query_expansions = {
            'ai': 'artificial intelligence AI',
            'ml': 'machine learning ML',
            'ocr': 'optical character recognition OCR document processing',
            'nlp': 'natural language processing NLP text analysis',
            'cv': 'computer vision CV image processing',
            'api': 'application programming interface API REST',
            'ui': 'user interface UI frontend',
            'db': 'database DB data storage'
        }
        
        for abbr, expansion in query_expansions.items():
            if abbr.lower() in query.lower():
                enhanced_query += f" {expansion}"
        
        docs = vectorstore.similarity_search(enhanced_query, k=k)
        logger.info(f"📚 Retrieved {len(docs)} documents for enhanced query: '{query}'")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []


def get_relevant_context(vectorstore, query, max_context_length=4000):
    """Enhanced context retrieval with comprehensive keywords from Parth's resume"""
    try:
        # Primary search with more results
        primary_docs = search_knowledge_base(vectorstore, query, k=8)
        
        # Comprehensive keyword mapping based on your resume
        query_lower = query.lower()
        secondary_queries = []
        
        # WORK EXPERIENCE KEYWORDS
        if any(term in query_lower for term in ['experience', 'work', 'professional', 'background', 'intern', 'job']):
            secondary_queries.extend([
                'L&T Larsen Toubro AI intern January 2025',
                'AlgoBrain AI ML intern June 2024',
                'CRNN OCR 32% accuracy improvement',
                'face recognition 94% accuracy EfficientNet',
                'Power BI dashboards KPI visualization',
                'Flask REST API Shopify integration'
            ])
        
        # PROJECT KEYWORDS - All 7 projects
        if any(term in query_lower for term in ['project', 'built', 'developed', 'created', 'system', 'application']):
            secondary_queries.extend([
                'Intelligent Web Scraping Agentic AI GPT-4 Playwright',
                'Advanced OCR Document Processing CRNN GRU',
                'Neurodiagnostic AI Assistant CNN DenseNet VGG16',
                'Face Recognition System TensorFlow EfficientNet',
                'Sentiment Analysis Pipeline BERT LSTM RNN',
                'Neural Network from Scratch NumPy MNIST',
                'Voice RAG Chatbot Streamlit Gemini ChromaDB 100x'
            ])
        
        # TECHNICAL SKILLS KEYWORDS
        if any(term in query_lower for term in ['skill', 'technical', 'technology', 'programming', 'language', 'framework']):
            secondary_queries.extend([
                'Python SQL JavaScript Bash programming',
                'PyTorch TensorFlow Keras Scikit-learn',
                'OpenCV CRNN LSTM CTC Loss BERT GPT-4',
                'FastAPI Flask Streamlit Docker RESTful APIs',
                'MySQL MongoDB ChromaDB Pinecone FAISS',
                'LangGraph AGNO Multi-Agent Architectures',
                'Power BI Tableau Git VS Code AWS'
            ])
        
        # AI/ML SPECIFIC KEYWORDS
        if any(term in query_lower for term in ['ai', 'ml', 'machine learning', 'artificial intelligence', 'deep learning', 'neural network']):
            secondary_queries.extend([
                'supervised learning unsupervised learning CNN RNN',
                'computer vision NLP natural language processing',
                'agentic AI LLM implementation GPU optimization',
                'TensorFlow Lite model optimization inference',
                'statistical analysis algorithm development MLOps'
            ])
        
        # ACHIEVEMENT & METRICS KEYWORDS
        if any(term in query_lower for term in ['achievement', 'result', 'accuracy', 'improvement', 'metric', 'performance']):
            secondary_queries.extend([
                '32% accuracy improvement OCR model',
                '94% accuracy face recognition system',
                '90% accuracy tumor classification',
                '84% accuracy MNIST neural network',
                '40% training time reduction GPU acceleration',
                '60% response time reduction chatbot',
                '40% user engagement increase recommendation',
                '1000+ documents daily processing pipeline'
            ])
        
        # EDUCATION & ACADEMIC KEYWORDS
        if any(term in query_lower for term in ['education', 'college', 'degree', 'study', 'academic', 'graduation']):
            secondary_queries.extend([
                'B.Tech Artificial Intelligence Data Science',
                'Sarvajanik College Engineering Technology Surat Gujarat',
                '2021 2025 graduation AI DS specialization'
            ])
        
        # PERSONAL CHARACTERISTICS KEYWORDS
        if any(term in query_lower for term in ['superpower', 'strength', 'ability', 'skill', 'talent']):
            secondary_queries.extend([
                'rapid learning adaptation superpower',
                'persistent curious disciplined practice',
                'connecting dots fast practical solutions'
            ])
        
        # GROWTH & DEVELOPMENT KEYWORDS
        if any(term in query_lower for term in ['growth', 'improve', 'develop', 'learn', 'future', 'goal']):
            secondary_queries.extend([
                'leadership team management mentor developers',
                'system architecture distributed systems microservices',
                'product strategy business value user problems'
            ])
        
        # COMPANY & CAREER KEYWORDS
        if any(term in query_lower for term in ['100x', 'company', 'why', 'career', 'future', 'remote']):
            secondary_queries.extend([
                '100x AI agents automation remote-first innovation',
                'cutting-edge technology breakthrough innovation',
                'autonomous systems human capabilities enhancement'
            ])
        
        # SPECIFIC TECHNOLOGIES KEYWORDS
        if any(term in query_lower for term in ['pytorch', 'tensorflow', 'python', 'opencv', 'bert', 'gpt']):
            secondary_queries.extend([
                'PyTorch dynamic research experimentation',
                'TensorFlow production deployment ecosystem',
                'Python primary programming language expertise',
                'OpenCV computer vision image processing',
                'BERT transformer NLP sentiment analysis',
                'GPT-4 LLM agentic systems automation'
            ])
        
        # PROBLEM SOLVING & METHODOLOGY KEYWORDS
        if any(term in query_lower for term in ['approach', 'solve', 'method', 'process', 'challenge', 'problem']):
            secondary_queries.extend([
                'iterative development minimum viable solution',
                'break down manageable components understanding requirements',
                'experimentation research debugging persistent curiosity'
            ])
        
        # Gather additional context from secondary searches
        all_docs = primary_docs[:]
        for secondary_query in secondary_queries[:4]:  # Limit to avoid context overflow
            additional_docs = search_knowledge_base(vectorstore, secondary_query, k=2)
            all_docs.extend(additional_docs)
        
        # Remove duplicates and build context
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_key = doc.page_content.strip()[:150]  # Use first 150 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)
        
        # Build context prioritizing the most relevant content
        context_parts = []
        total_length = 0
        
        # Prioritize different content types based on query
        if any(term in query_lower for term in ['project', 'built', 'developed']):
            # Prioritize project content
            project_docs = [doc for doc in unique_docs if any(term in doc.page_content.lower() 
                           for term in ['project', 'built', 'developed', 'system', 'model', 'pipeline'])]
            other_docs = [doc for doc in unique_docs if doc not in project_docs]
            ordered_docs = project_docs + other_docs
        elif any(term in query_lower for term in ['experience', 'work', 'intern']):
            # Prioritize work experience content
            work_docs = [doc for doc in unique_docs if any(term in doc.page_content.lower() 
                        for term in ['intern', 'l&t', 'algobrain', 'work experience'])]
            other_docs = [doc for doc in unique_docs if doc not in work_docs]
            ordered_docs = work_docs + other_docs
        else:
            ordered_docs = unique_docs
        
        # Build final context
        for doc in ordered_docs:
            content = doc.page_content.strip()
            if total_length + len(content) <= max_context_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                remaining_space = max_context_length - total_length
                if remaining_space > 200:
                    context_parts.append(content[:remaining_space] + "...")
                break
        
        final_context = "\n\n".join(context_parts)
        logger.info(f"📋 Built context: {len(final_context)} chars from {len(context_parts)} chunks with {len(secondary_queries)} secondary searches")
        
        return final_context
        
    except Exception as e:
        logger.error(f"❌ Context retrieval error: {e}")
        return "Context temporarily unavailable - using basic information."

