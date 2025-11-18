import streamlit as st
import asyncio
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import tempfile
import shutil
from datetime import datetime
import threading

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
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

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def load_pdf_to_vectordb(pdf_path: str, session_id: str):
    """Load PDF, split into chunks, embed and store in ChromaDB"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No content found in PDF")
    
    # Combine all text from pages
    total_text = " ".join([doc.page_content for doc in documents if doc.page_content.strip()])
    
    if not total_text.strip():
        raise ValueError("PDF appears to be empty or contains no readable text")
    
    # Split documents into chunks - be more flexible with size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for flexibility
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )
    
    # If document is very short, just use it as is
    if len(total_text) < 500:
        chunks = documents
    else:
        chunks = text_splitter.split_documents(documents)
    
    # Filter out empty chunks and very short ones
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 20]
    
    if not chunks:
        raise ValueError(f"Could not create valid chunks. PDF has {len(total_text)} characters but failed to split properly.")
    
    # Create vector database for this session
    persist_dir = f"./chroma_db_{session_id}"
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=st.session_state.embeddings,
        persist_directory=persist_dir
    )
    
    return vectordb, len(chunks)

def search_vectordb_factory(vectordb_ref: dict):
    """Factory function to create search function with vectordb reference"""
    def search_vectordb(query: str, k: int = 3) -> str:
        """Search the vector database for relevant content based on a query. Returns the top 3 most relevant text chunks from the loaded PDF documents. Use specific keywords for better results."""
        vectordb = vectordb_ref.get('db')
        
        if vectordb is None:
            return "Error: No PDF loaded in this session. Please upload a PDF first."
        
        # Reduced k to 3 for better token efficiency
        results = vectordb.similarity_search(query, k=k)
        
        if not results:
            return "No relevant content found in the database."
        
        # Format results more concisely - limit each chunk display
        formatted_results = []
        for i, doc in enumerate(results, 1):
            # Limit content length to avoid token overflow
            content = doc.page_content
            if len(content) > 800:  # Limit chunk size
                content = content[:800] + "..."
            formatted_results.append(f"[Chunk {i}]: {content}")
        
        return "\n\n".join(formatted_results)
    
    return search_vectordb

def create_agents_for_session(vectordb_ref: dict):
    """Create agents with session-specific tools"""
    db_search_tool = FunctionTool(func=search_vectordb_factory(vectordb_ref))
    
    # Simple single agent with just PDF search
    main_agent = Agent(
        name="pdf_chat_agent",
        model=Gemini(
            model="gemini-2.0-flash-exp",
            retry_options=retry_config
        ),
        instruction="""You are a PDF document assistant. 

KEY RULES:
1. For ANY document question, ALWAYS call search_vectordb first with relevant keywords
2. Use targeted search terms (e.g., "summary main points" for summaries, "introduction" for intros)
3. Keep responses concise and focused on the retrieved content
4. Answer based only on the PDF content retrieved

Examples:
- "summarize" → search_vectordb(query="main points summary key information")
- "what is X" → search_vectordb(query="X definition explanation")
- General questions → search_vectordb(query="document overview content")""",
        tools=[db_search_tool]
    )
    
    return main_agent

def create_new_session(session_name: str):
    """Create a new chat session"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create a mutable reference for vectordb that can be updated
    vectordb_ref = {'db': None}
    
    st.session_state.sessions[session_id] = {
        'name': session_name,
        'messages': [],
        'vectordb': None,
        'vectordb_ref': vectordb_ref,  # Store the reference
        'pdf_name': None
    }
    st.session_state.current_session_id = session_id
    return session_id

async def get_agent_response(session_data: dict, user_message: str):
    """Get response using manual search + direct model call (no agents/tools)"""
    from google import genai
    import os
    
    # Step 1: Search vector database manually for relevant context
    vectordb = session_data['vectordb_ref'].get('db')
    
    if vectordb is None:
        return "Please upload a PDF document first to start chatting."
    
    # Perform similarity search to get relevant chunks
    results = vectordb.similarity_search(user_message, k=3)
    
    if not results:
        context = "No relevant content found in the PDF."
    else:
        # Format the retrieved chunks as context
        context_parts = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            if len(content) > 800:
                content = content[:800] + "..."
            context_parts.append(f"[Excerpt {i}]:\n{content}")
        context = "\n\n".join(context_parts)
    
    # Step 2: Create prompt with context and user question
    prompt = f"""You are a PDF document assistant. Answer the user's question based ONLY on the following excerpts from the PDF document.

PDF EXCERPTS:
{context}

USER QUESTION: {user_message}

INSTRUCTIONS:
- Answer based only on the excerpts provided above
- Be concise and specific
- If the excerpts don't contain relevant information, say so
- Do not make up information

ANSWER:"""
    
    # Step 3: Call model directly (no agent, no tools - just text generation)
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt
    )
    
    # Extract response text
    response_text = ""
    if hasattr(response, 'text'):
        response_text = response.text
    elif hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
    
    return response_text if response_text else "I couldn't generate a response. Please try again."

def run_async_in_thread(coro):
    """Run async function in a new thread with its own event loop"""
    result = [None]
    exception = [None]
    
    def run_in_thread():
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(coro)
            
            # Comprehensive cleanup to prevent "Event loop is closed" errors
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for task cancellations
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Shutdown async generators
            loop.run_until_complete(loop.shutdown_asyncgens())
            
            # Give aiohttp and other async resources time to cleanup
            loop.run_until_complete(asyncio.sleep(0.2))
            
        except Exception as e:
            exception[0] = e
        finally:
            # Close the loop in finally block to ensure it happens
            if loop is not None and not loop.is_closed():
                loop.close()
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

# Sidebar for session management
with st.sidebar:
    st.title("📚 PDF Chat Assistant")
    st.markdown("---")
    
    # New Session
    st.subheader("Create New Session")
    new_session_name = st.text_input("Session Name", placeholder="My PDF Chat")
    if st.button("➕ New Session", use_container_width=True):
        if new_session_name:
            create_new_session(new_session_name)
            st.success(f"Created session: {new_session_name}")
            st.rerun()
        else:
            st.error("Please enter a session name")
    
    st.markdown("---")
    
    # Session Selection
    st.subheader("Active Sessions")
    if st.session_state.sessions:
        session_options = {
            session_id: f"{data['name']} {'📄' if data['pdf_name'] else '❌'}"
            for session_id, data in st.session_state.sessions.items()
        }
        
        selected_session = st.selectbox(
            "Select Session",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=list(session_options.keys()).index(st.session_state.current_session_id) 
                if st.session_state.current_session_id in session_options else 0
        )
        st.session_state.current_session_id = selected_session
        
        # Delete session button
        if st.button("🗑️ Delete Session", use_container_width=True):
            session_id = st.session_state.current_session_id
            # Clean up vector database
            persist_dir = f"./chroma_db_{session_id}"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            del st.session_state.sessions[session_id]
            st.session_state.current_session_id = None
            st.rerun()
    else:
        st.info("No sessions yet. Create one to start!")
    
    st.markdown("---")

# Main chat interface
if st.session_state.current_session_id:
    current_session = st.session_state.sessions[st.session_state.current_session_id]
    
    st.title(f"💬 {current_session['name']}")
    
    # PDF Upload section at the top of chat
    with st.expander("📄 Upload PDF for this chat", expanded=not current_session['pdf_name']):
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key=f"uploader_{st.session_state.current_session_id}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if current_session['pdf_name']:
                st.success(f"✅ Currently loaded: **{current_session['pdf_name']}**")
            else:
                st.info("No PDF loaded yet. Upload one to enable document search.")
        
        with col2:
            if uploaded_file is not None:
                if st.button("📤 Process PDF", use_container_width=True):
                    with st.spinner("Processing PDF..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Load PDF to vector database
                            vectordb, num_chunks = load_pdf_to_vectordb(
                                tmp_path, 
                                st.session_state.current_session_id
                            )
                            
                            # Update session and the reference
                            current_session['vectordb'] = vectordb
                            current_session['vectordb_ref']['db'] = vectordb  # Update the reference
                            current_session['pdf_name'] = uploaded_file.name
                            
                            st.success(f"✅ Processed {num_chunks} chunks from {uploaded_file.name}!")
                            st.balloons()
                            st.rerun()
                        except ValueError as ve:
                            st.error(f"❌ PDF Processing Error: {ve}")
                        except Exception as e:
                            st.error(f"❌ Unexpected Error: {str(e)}")
                            st.info("Try uploading a different PDF or check if the file is valid.")
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
    
    st.markdown("---")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in current_session['messages']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        current_session['messages'].append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_async_in_thread(get_agent_response(current_session, prompt))
                st.markdown(response)
        
        # Add assistant message
        current_session['messages'].append({'role': 'assistant', 'content': response})
        
        st.rerun()
    
    # Clear chat button
    if len(current_session['messages']) > 0:
        if st.button("🗑️ Clear Chat History"):
            current_session['messages'] = []
            st.rerun()

else:
    st.title("Welcome to PDF Chat Assistant! 📚")
    st.markdown("""
    ### Getting Started
    
    1. **Create a Session** - Use the sidebar to create a new chat session
    2. **Upload a PDF** - Upload your document to analyze
    3. **Start Chatting** - Ask questions about your PDF!
    
    ### Features
    - 📄 Upload and analyze PDF documents
    - 💬 Interactive chat interface
    - 🔍 Semantic search through your documents
    - 📝 Generate summaries and questions
    - 🔗 Google search integration for additional context
    - 💾 Multiple session management
    """)
