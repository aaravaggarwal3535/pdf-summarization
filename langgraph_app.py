"""
alternative of the google adk app in langchain made to make better comparision in both and what is better and the results show that the code is much more easy to write and manage in google-adk
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal
import tempfile
import shutil
from datetime import datetime
from ddgs import DDGS

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant - Google ADK",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the state for LangGraph
class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    vectordb: any
    search_type: str  # 'pdf', 'web', or 'both'
    web_results: str

def load_pdf_to_vectordb(pdf_path: str, session_id: str):
    """Load PDF, split into chunks, embed and store in ChromaDB"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    if not documents:
        st.error("Failed to load PDF documents")
        return None, 0
    
    # Split into chunks (smaller chunks for token efficiency)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not chunks:
        st.error("No valid content found in PDF")
        return None, 0
    
    # Create embeddings and store in ChromaDB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=st.session_state.embeddings,
        persist_directory=f"./chroma_db_{session_id}"
    )
    
    return vectordb, len(chunks)

# LangGraph node functions
def route_question(state: ChatState) -> ChatState:
    """Determine if question is about PDF, web search, or both"""
    question = state.get('question', '').lower()
    
    # Keywords that suggest web search
    web_keywords = ['search', 'google', 'duckduckgo', 'dugduggo', 'find online', 'lookup', 'does exist', 'is there', 'are there', 'exist', 'real world', 'actual']
    
    # Keywords that suggest PDF only
    pdf_only_keywords = ['summarize', 'summary', 'tell me about', 'what does', 'explain', 'describe']
    
    # Check for explicit web search request
    wants_web = any(keyword in question for keyword in web_keywords)
    
    # If asking about PDF content first, then wants web search after
    if wants_web:
        state['search_type'] = 'web'
    elif any(keyword in question for keyword in pdf_only_keywords) and 'search' not in question:
        state['search_type'] = 'pdf'
    else:
        # Default to PDF if available, otherwise web
        state['search_type'] = 'pdf' if state.get('vectordb') is not None else 'web'
    
    return state

def retrieve_context(state: ChatState) -> ChatState:
    """Retrieve relevant context from vector database"""
    vectordb = state.get('vectordb')
    question = state.get('question')
    
    if vectordb is None:
        state['context'] = ""
        return state
    
    # Perform similarity search
    results = vectordb.similarity_search(question, k=3)
    
    if not results:
        state['context'] = ""
        return state
    
    # Format context
    context_parts = []
    for i, doc in enumerate(results, 1):
        content = doc.page_content
        if len(content) > 800:
            content = content[:800] + "..."
        context_parts.append(f"[PDF Excerpt {i}]:\n{content}")
    
    state['context'] = "\n\n".join(context_parts)
    return state

def web_search(state: ChatState) -> ChatState:
    """Perform DuckDuckGo web search"""
    question = state.get('question')
    
    try:
        # Use the new ddgs package
        ddgs = DDGS()
        results = ddgs.text(question, max_results=5)
        
        # Format results
        if results:
            formatted_results = []
            for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                formatted_results.append(f"Result {i}: {title}\n{body}")
            
            state['web_results'] = f"[Web Search Results]:\n" + "\n\n".join(formatted_results)
        else:
            state['web_results'] = "No web search results found."
    except Exception as e:
        state['web_results'] = f"Web search failed: {str(e)}"
    
    return state

def generate_answer(state: ChatState) -> ChatState:
    """Generate answer using LLM with context"""
    context = state.get('context', '')
    web_results = state.get('web_results', '')
    question = state.get('question', '')
    search_type = state.get('search_type', 'pdf')
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.3
    )
    
    # Create prompt based on search type
    if search_type == 'web':
        prompt = f"""You are a helpful assistant. Answer the user's question based on the following web search results.

WEB SEARCH RESULTS:
{web_results if web_results else "No web results available."}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based on the web search results provided
- Be concise and informative
- If the results don't contain relevant information, say so

ANSWER:"""
    
    elif search_type == 'pdf':
        prompt = f"""You are a PDF document assistant. Answer the user's question based ONLY on the following excerpts from the PDF document.

PDF EXCERPTS:
{context if context else "No PDF content available. Please upload a PDF first."}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the PDF excerpts provided above
- Be concise and specific
- If the excerpts don't contain relevant information, say so
- Do not make up information

ANSWER:"""
    
    else:  # both
        prompt = f"""You are a helpful assistant with access to both PDF documents and web search results. Answer the user's question using the available information.

PDF EXCERPTS:
{context if context else "No PDF available."}

WEB SEARCH RESULTS:
{web_results if web_results else "No web results available."}

USER QUESTION: {question}

INSTRUCTIONS:
- Use information from both sources when available
- Cite your sources (PDF or Web)
- Be concise and accurate

ANSWER:"""
    
    # Generate response
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        state['answer'] = response.content
    except Exception as e:
        state['answer'] = f"Error generating response: {str(e)}"
    
    return state

# Build LangGraph workflow
def create_workflow():
    """Create the LangGraph workflow with conditional routing"""
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("route", route_question)
    workflow.add_node("retrieve_pdf", retrieve_context)
    workflow.add_node("search_web", web_search)
    workflow.add_node("generate", generate_answer)
    
    # Define routing function
    def route_to_search(state: ChatState) -> str:
        search_type = state.get('search_type', 'pdf')
        if search_type == 'web':
            return "search_web"
        elif search_type == 'pdf':
            return "retrieve_pdf"
        else:
            return "both"
    
    # Add edges
    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        route_to_search,
        {
            "retrieve_pdf": "retrieve_pdf",
            "search_web": "search_web",
            "both": "retrieve_pdf"  # If both, do PDF first then web
        }
    )
    workflow.add_edge("retrieve_pdf", "generate")
    workflow.add_edge("search_web", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def create_new_session(session_name: str):
    """Create a new chat session"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    st.session_state.sessions[session_id] = {
        'name': session_name,
        'messages': [],
        'vectordb': None,
        'pdf_name': None
    }
    st.session_state.current_session_id = session_id
    return session_id

# Streamlit UI
st.title("📚 PDF Chat Assistant (Google ADK)")

# Sidebar
with st.sidebar:
    st.header("Chat Sessions")
    
    # New session
    with st.expander("➕ New Session"):
        new_session_name = st.text_input("Session Name", key="new_session_input")
        if st.button("Create Session"):
            if new_session_name:
                create_new_session(new_session_name)
                st.success(f"Created session: {new_session_name}")
                st.rerun()
            else:
                st.warning("Please enter a session name")
    
    # List sessions
    if st.session_state.sessions:
        for session_id, session_data in st.session_state.sessions.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    f"📁 {session_data['name']}", 
                    key=f"btn_{session_id}",
                    type="primary" if session_id == st.session_state.current_session_id else "secondary"
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{session_id}"):
                    # Delete session
                    if session_data['vectordb'] is not None:
                        db_dir = f"./chroma_db_{session_id}"
                        if os.path.exists(db_dir):
                            shutil.rmtree(db_dir)
                    del st.session_state.sessions[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    st.rerun()
    else:
        st.info("No sessions yet. Create one to start!")

# Main area
if st.session_state.current_session_id:
    current_session = st.session_state.sessions[st.session_state.current_session_id]
    
    st.subheader(f"Session: {current_session['name']}")
    
    # PDF Upload
    uploaded_file = st.file_uploader(
        "Upload PDF for this session",
        type=['pdf'],
        key=f"upload_{st.session_state.current_session_id}"
    )
    
    if uploaded_file:
        if current_session['pdf_name'] != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Load and process PDF
                    vectordb, num_chunks = load_pdf_to_vectordb(
                        tmp_path,
                        st.session_state.current_session_id
                    )
                    
                    if vectordb:
                        current_session['vectordb'] = vectordb
                        current_session['pdf_name'] = uploaded_file.name
                        st.success(f"✅ Processed {num_chunks} chunks from {uploaded_file.name}")
                        st.balloons()
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    if current_session['pdf_name']:
        st.info(f"📄 Current PDF: {current_session['pdf_name']}")
    
    # Chat interface
    st.divider()
    
    # Display chat history
    for message in current_session['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your PDF or search the web..."):
        # Add user message to chat
        current_session['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create workflow
                app = create_workflow()
                
                # Run workflow
                initial_state = {
                    "question": prompt,
                    "context": "",
                    "answer": "",
                    "vectordb": current_session['vectordb'],
                    "search_type": "",
                    "web_results": ""
                }
                
                result = app.invoke(initial_state)
                response = result.get('answer', "I couldn't generate a response. Please try again.")
                
                st.markdown(response)
        
        # Add assistant message to chat
        current_session['messages'].append({"role": "assistant", "content": response})
        st.rerun()

else:
    st.info("👈 Create or select a session from the sidebar to start chatting with your PDFs!")
    
    # Show some helpful information
    st.markdown("""
    ### How to use:
    1. Create a new session from the sidebar
    2. Upload a PDF document (optional)
    3. Ask questions about the PDF content or search the web
    4. The system will automatically route to PDF search or web search
    
    ### Features:
    - ✅ Multiple chat sessions
    - ✅ Session-specific PDF uploads
    - ✅ Efficient embedding-based search (only relevant chunks)
    - ✅ DuckDuckGo web search integration
    - ✅ Intelligent routing (PDF vs Web)
    - ✅ Google ADK workflow for structured processing
    - ✅ Powered by Gemini 2.0 Flash
    
    ### Example queries:
    - **PDF**: "Summarize the document", "What does section 3 say?"
    - **Web**: "Search for XYZ company", "Find information about ABC startup"
    """)
