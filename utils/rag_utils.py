# utils/rag_utils.py
import os
import streamlit as st
import pickle
import json
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("SentenceTransformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class SimpleDocument:
    """Simple document class for when LangChain is not available"""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class DeploymentFriendlyRAGSystem:
    """RAG system using FAISS + SentenceTransformers for deployment compatibility"""
    
    def __init__(self, file_path="knowledge_base/personal_info.txt", persist_dir="vectorstore_data"):
        self.file_path = file_path
        self.persist_dir = persist_dir
        self.embeddings_model = None
        self.index = None
        self.documents = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
    def initialize_embeddings(self):
        """Initialize SentenceTransformer embeddings model"""
        try:
            if not EMBEDDINGS_AVAILABLE:
                logger.error("SentenceTransformers not available")
                return False
                
            self.embeddings_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cpu'
            )
            logger.info("✅ SentenceTransformer model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings model: {e}")
            return False
    
    def load_personal_document(self):
        """Load Parth's personal_info.txt document"""
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                # Create a fallback document with basic info
                fallback_content = self.get_fallback_content()
                logger.warning(f"⚠️ {self.file_path} not found, using fallback content")
                return [SimpleDocument(fallback_content, {"source": "fallback"})]
            
            # Load the file
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            documents = [SimpleDocument(content, {"source": self.file_path})]
            logger.info(f"✅ Loaded personal_info.txt successfully")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading personal_info.txt: {e}")
            fallback_content = self.get_fallback_content()
            return [SimpleDocument(fallback_content, {"source": "fallback"})]
    
    def get_fallback_content(self):
        """Fallback content when personal_info.txt is not available"""
        return """
PARTH PATEL - AI/ML ENGINEER

PROFESSIONAL EXPERIENCE:
- AI Intern at L&T (Jan 2025 - Apr 2025): Developed CRNN-based OCR system achieving 32% accuracy improvement, processed 1000+ documents daily, created Power BI dashboards for KPI visualization
- ML Intern at AlgoBrain AI (Jun 2024 - Jul 2024): Built face recognition system with 94% accuracy using EfficientNet, developed Flask REST APIs, integrated Shopify systems

EDUCATION:
- B.Tech in Artificial Intelligence & Data Science (2021-2025)
- Sarvajanik College of Engineering & Technology, Surat, Gujarat

KEY PROJECTS:
1. Intelligent Web Scraping with Agentic AI: GPT-4 powered automation using Playwright
2. Advanced OCR Document Processing: CRNN with GRU achieving 32% improvement
3. Neurodiagnostic AI Assistant: CNN models (DenseNet, VGG16) for 90%+ tumor classification
4. Face Recognition System: TensorFlow EfficientNet with 94% accuracy
5. Sentiment Analysis Pipeline: BERT and LSTM models for text analysis
6. Neural Network from Scratch: NumPy implementation achieving 84% MNIST accuracy
7. Voice RAG Chatbot: Current project using Streamlit, Gemini, and vector databases

TECHNICAL SKILLS:
- Programming: Python, SQL, JavaScript, Bash
- AI/ML: PyTorch, TensorFlow, Keras, Scikit-learn, OpenCV
- NLP: BERT, GPT-4, LSTM, Transformers, LangGraph
- Databases: MySQL, MongoDB, ChromaDB, Pinecone, FAISS
- Tools: Docker, FastAPI, Flask, Streamlit, Power BI, Git, AWS

ACHIEVEMENTS:
- 32% accuracy improvement in OCR systems
- 94% accuracy in face recognition
- 90%+ accuracy in medical image classification
- 1000+ documents processing pipeline
- Multiple published technical articles

CAREER INTERESTS:
Interested in 100x for AI agents, automation, and remote-first innovation culture. Focus areas: rapid learning, system architecture, and product strategy.
"""
    
    def process_documents(self, documents):
        """Split documents into optimized chunks"""
        if not documents:
            logger.warning("⚠️ No documents to process")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for better precision
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "**", "- ", ". ", " ", ""]
        )
        
        all_chunks = []
        for doc in documents:
            # Split the document content
            chunks = text_splitter.split_text(doc.page_content)
            
            # Create document objects for each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = SimpleDocument(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"✅ Created {len(all_chunks)} chunks from documents")
        return all_chunks
    
    def create_vectorstore(self, doc_chunks):
        """Create FAISS vectorstore from document chunks"""
        try:
            if not self.embeddings_model:
                if not self.initialize_embeddings():
                    raise Exception("Failed to initialize embeddings model")
            
            if not FAISS_AVAILABLE:
                raise Exception("FAISS not available")
            
            if not doc_chunks:
                raise Exception("No document chunks available")
            
            # Extract text content and generate embeddings
            texts = [doc.page_content for doc in doc_chunks]
            embeddings = self.embeddings_model.encode(texts, convert_to_numpy=True)
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            # Store documents for retrieval
            self.documents = doc_chunks
            
            # Save vectorstore to disk
            self.save_vectorstore()
            
            logger.info(f"✅ Created FAISS vectorstore with {len(doc_chunks)} documents")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            return None
    
    def save_vectorstore(self):
        """Save vectorstore to disk"""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss_index.bin"))
            
            # Save documents
            docs_data = []
            for doc in self.documents:
                docs_data.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            with open(os.path.join(self.persist_dir, "documents.json"), 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "embedding_dim": self.embedding_dim,
                "num_documents": len(self.documents)
            }
            
            with open(os.path.join(self.persist_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Vectorstore saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving vectorstore: {e}")
    
    def load_vectorstore(self):
        """Load existing vectorstore from disk"""
        try:
            if not os.path.exists(self.persist_dir):
                return False
            
            # Load FAISS index
            index_path = os.path.join(self.persist_dir, "faiss_index.bin")
            if os.path.exists(index_path) and FAISS_AVAILABLE:
                self.index = faiss.read_index(index_path)
            else:
                return False
            
            # Load documents
            docs_path = os.path.join(self.persist_dir, "documents.json")
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                
                self.documents = []
                for doc_data in docs_data:
                    doc = SimpleDocument(
                        page_content=doc_data["page_content"],
                        metadata=doc_data["metadata"]
                    )
                    self.documents.append(doc)
            else:
                return False
            
            # Initialize embeddings model
            if not self.initialize_embeddings():
                return False
            
            logger.info(f"✅ Loaded existing vectorstore with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading existing vectorstore: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5):
        """Search for similar documents"""
        try:
            if not self.embeddings_model or not self.index:
                logger.error("❌ Vectorstore not properly initialized")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return matching documents
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]
                    results.append(doc)
            
            logger.info(f"📚 Retrieved {len(results)} documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

# Fallback simple search when vector search is not available
class FallbackRAGSystem:
    """Simple keyword-based search when vector embeddings are not available"""
    
    def __init__(self, file_path="knowledge_base/personal_info.txt"):
        self.file_path = file_path
        self.content = ""
        self.load_content()
    
    def load_content(self):
        """Load content for keyword search"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            else:
                # Use fallback content
                rag_system = DeploymentFriendlyRAGSystem()
                self.content = rag_system.get_fallback_content()
                
            logger.info("✅ Loaded content for fallback search")
        except Exception as e:
            logger.error(f"❌ Error loading content: {e}")
            # Use minimal fallback
            self.content = "Parth Patel - AI/ML Engineer with experience at L&T and AlgoBrain AI"
    
    def similarity_search(self, query: str, k: int = 5):
        """Simple keyword-based search"""
        try:
            # Split content into chunks
            chunks = []
            sentences = self.content.split('\n')
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + "\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "\n"
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Score chunks based on keyword overlap
            query_words = set(query.lower().split())
            scored_chunks = []
            
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                score = len(query_words.intersection(chunk_words))
                if score > 0:
                    doc = SimpleDocument(chunk, {"score": score})
                    scored_chunks.append((score, doc))
            
            # Sort by score and return top k
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            results = [doc for score, doc in scored_chunks[:k]]
            
            logger.info(f"📚 Fallback search found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fallback search error: {e}")
            # Return basic info
            return [SimpleDocument("Parth Patel - AI/ML Engineer", {})]

@st.cache_resource
def load_vectorstore(file_path="knowledge_base/personal_info.txt", persist_dir="vectorstore_data"):
    """Load vectorstore with multiple fallback strategies"""
    try:
        # Try FAISS-based RAG system first
        if EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
            rag_system = DeploymentFriendlyRAGSystem(file_path, persist_dir)
            
            # Try to load existing vectorstore
            if rag_system.load_vectorstore():
                logger.info("✅ Loaded existing FAISS vectorstore")
                return rag_system
            
            # Create new vectorstore
            documents = rag_system.load_personal_document()
            if documents:
                doc_chunks = rag_system.process_documents(documents)
                if doc_chunks:
                    vectorstore = rag_system.create_vectorstore(doc_chunks)
                    if vectorstore:
                        logger.info("✅ Created new FAISS vectorstore")
                        return vectorstore
        
        # Fall back to simple keyword search
        logger.warning("⚠️ Using fallback keyword search")
        return FallbackRAGSystem(file_path)
        
    except Exception as e:
        logger.error(f"❌ Error in load_vectorstore: {e}")
        # Ultimate fallback
        return FallbackRAGSystem(file_path)

def search_knowledge_base(vectorstore, query, k=5):
    """Search knowledge base with error handling"""
    try:
        if vectorstore is None:
            logger.error("❌ Vectorstore is None")
            return []
        
        # Enhanced query preprocessing
        enhanced_query = query
        
        # Query expansions for better matching
        query_expansions = {
            'ai': 'artificial intelligence AI machine learning',
            'ml': 'machine learning ML artificial intelligence',
            'ocr': 'optical character recognition OCR document processing CRNN',
            'nlp': 'natural language processing NLP text analysis BERT',
            'cv': 'computer vision CV image processing OpenCV',
            'project': 'project built developed created system application',
            'experience': 'experience work intern professional background',
            'skill': 'skill technical technology programming language framework'
        }
        
        for abbr, expansion in query_expansions.items():
            if abbr.lower() in query.lower():
                enhanced_query += f" {expansion}"
        
        docs = vectorstore.similarity_search(enhanced_query, k=k)
        return docs
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []

def get_relevant_context(vectorstore, query, max_context_length=4000):
    """Get relevant context with comprehensive keyword matching"""
    try:
        # Primary search
        primary_docs = search_knowledge_base(vectorstore, query, k=6)
        
        # Enhanced secondary searches based on query type
        query_lower = query.lower()
        secondary_queries = []
        
        # Work experience keywords
        if any(term in query_lower for term in ['experience', 'work', 'professional', 'background', 'intern']):
            secondary_queries.extend([
                'L&T Larsen Toubro AI intern',
                'AlgoBrain AI ML intern',
                'CRNN OCR accuracy improvement',
                'face recognition EfficientNet',
                'Power BI dashboards'
            ])
        
        # Project keywords
        if any(term in query_lower for term in ['project', 'built', 'developed', 'system']):
            secondary_queries.extend([
                'Web Scraping GPT-4 Playwright',
                'OCR Document Processing CRNN',
                'Neurodiagnostic AI CNN',
                'Face Recognition TensorFlow',
                'Sentiment Analysis BERT',
                'Neural Network NumPy MNIST',
                'Voice RAG Chatbot Streamlit'
            ])
        
        # Technical skills keywords
        if any(term in query_lower for term in ['skill', 'technical', 'technology', 'programming']):
            secondary_queries.extend([
                'Python PyTorch TensorFlow',
                'OpenCV BERT GPT-4',
                'FastAPI Flask Streamlit',
                'MySQL MongoDB ChromaDB',
                'Docker AWS Git'
            ])
        
        # Achievement keywords
        if any(term in query_lower for term in ['achievement', 'accuracy', 'improvement', 'result']):
            secondary_queries.extend([
                '32% accuracy improvement OCR',
                '94% accuracy face recognition',
                '90% accuracy tumor classification',
                '1000+ documents processing'
            ])
        
        # Career/company keywords
        if any(term in query_lower for term in ['100x', 'company', 'career', 'future']):
            secondary_queries.extend([
                '100x AI agents automation',
                'remote-first innovation culture',
                'rapid learning adaptation',
                'system architecture product strategy'
            ])
        
        # Collect additional context
        all_docs = primary_docs[:]
        for secondary_query in secondary_queries[:3]:  # Limit secondary searches
            additional_docs = search_knowledge_base(vectorstore, secondary_query, k=2)
            all_docs.extend(additional_docs)
        
        # Remove duplicates and build context
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_key = doc.page_content.strip()[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)
        
        # Build final context
        context_parts = []
        total_length = 0
        
        for doc in unique_docs:
            content = doc.page_content.strip()
            if total_length + len(content) <= max_context_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                remaining_space = max_context_length - total_length
                if remaining_space > 150:
                    context_parts.append(content[:remaining_space] + "...")
                break
        
        final_context = "\n\n".join(context_parts)
        logger.info(f"📋 Built context: {len(final_context)} chars from {len(context_parts)} chunks")
        
        return final_context if final_context else "Basic information available about Parth Patel's AI/ML background."
        
    except Exception as e:
        logger.error(f"❌ Context retrieval error: {e}")
        return "Parth Patel - AI/ML Engineer with experience in computer vision, NLP, and system development."
