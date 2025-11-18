# PDF Chat Assistant with Web Search

## Problem Statement

Modern knowledge work involves constantly switching between different information sources - reading PDFs, searching the web, cross-referencing documents, and synthesizing information from multiple places. This context-switching is time-consuming and breaks concentration. When working with technical documents, research papers, or business reports, users need to:

1. **Extract specific information** from lengthy PDF documents without reading everything
2. **Verify information** by cross-referencing with real-world data from web searches
3. **Maintain context** across multiple chat sessions with different documents
4. **Get accurate answers** based on actual document content, not hallucinated information

This is an important problem because:
- **Productivity**: Reduces time spent manually searching through documents
- **Accuracy**: Ensures answers are grounded in actual source material
- **Efficiency**: Single interface for both document Q&A and web research
- **Organization**: Multiple sessions allow working with different projects simultaneously

---

## Why Agents?

Agents are the right solution because this problem requires **intelligent routing and decision-making**:

### 1. **Dynamic Decision Making**
The system needs to decide:
- Should I search the PDF or the web?
- Which keywords to use for optimal retrieval?
- How to combine information from multiple sources?

### 2. **Multi-Step Reasoning**
The workflow involves:
- Understanding user intent
- Routing to appropriate data source
- Retrieving relevant context
- Synthesizing information into coherent answers

### 3. **Tool Orchestration**
Agents can seamlessly coordinate:
- Vector database searches (PDF embeddings)
- Web search APIs (DuckDuckGo)
- LLM generation (Gemini 2.0 Flash)

### 4. **Contextual Awareness**
Agents maintain state across the workflow:
- What PDF is loaded?
- What was asked previously?
- What context was retrieved?

Traditional search or simple RAG wouldn't provide the **intelligent routing** and **multi-source coordination** needed for this use case.

---

## What We Created

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  (Session Management, File Upload, Chat Interface)          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Workflow Engine                       │
│                                                              │
│  ┌──────────┐      ┌───────────────┐     ┌──────────────┐  │
│  │  Route   │─────▶│  Conditional  │────▶│   Generate   │  │
│  │  Node    │      │   Branching   │     │    Answer    │  │
│  └──────────┘      └───────┬───────┘     └──────────────┘  │
│                            │                                │
│                    ┌───────┴────────┐                       │
│                    ▼                ▼                        │
│           ┌─────────────┐  ┌──────────────┐                │
│           │ Retrieve    │  │ Web Search   │                │
│           │ PDF (VDB)   │  │ (DuckDuckGo) │                │
│           └─────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────┘
                  │                    │
                  ▼                    ▼
┌──────────────────────┐    ┌──────────────────────┐
│   ChromaDB           │    │   DuckDuckGo API     │
│   Vector Database    │    │   Web Search         │
│   (HuggingFace       │    │                      │
│    Embeddings)       │    │                      │
└──────────────────────┘    └──────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Gemini 2.0 Flash                         │
│         (Answer Generation with Context)                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Session Management**: Multiple isolated chat sessions, each with its own PDF and conversation history
2. **PDF Processing Pipeline**: 
   - PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings → ChromaDB
   - Chunks: 500 chars with 100 char overlap
   - Top-K retrieval: 3 most relevant chunks (max 800 chars each)
3. **Intelligent Router**: Analyzes user query for keywords to determine PDF vs Web search
4. **Dual Search Capabilities**:
   - **PDF Search**: Semantic similarity search using embeddings
   - **Web Search**: DuckDuckGo integration for real-time information
5. **Context-Aware Generation**: Gemini 2.0 Flash generates answers using only retrieved context

---

## Demo

### Screenshot / Video
[Your screenshot or demo video here]

### Example Interactions

**PDF Query:**
```
User: "Explain T-5 from the document"
Assistant: [Searches PDF] "T-5, titled 'Trust Is Broken – 2035', 
           focuses on rebuilding trust in a 5-star world where 
           people no longer believe fake reviews, polished photos, 
           or AI-generated testimonials..."
```

**Web Search Query:**
```
User: "Search if there exists a startup solving the T-5 trust problem"
Assistant: [Performs DuckDuckGo search] "Based on web search results, 
           several startups are working on trust and review verification, 
           including..."
```

**Session Features:**
- Upload different PDFs per session
- Switch between sessions seamlessly
- Persistent chat history per session
- Delete sessions and associated data

---

## The Build

### Technologies Used

#### Core Framework
- **LangGraph**: Workflow orchestration with conditional routing
- **LangChain**: Document processing, embeddings, LLM integration
- **Streamlit**: Interactive web UI with session management

#### AI & ML
- **Google Gemini 2.0 Flash**: LLM for answer generation
- **HuggingFace Transformers**: Sentence embeddings (all-MiniLM-L6-v2)
- **ChromaDB**: Vector database for semantic search

#### Document Processing
- **PyPDFLoader**: PDF text extraction
- **RecursiveCharacterTextSplitter**: Smart text chunking

#### Search & Retrieval
- **DuckDuckGo (ddgs)**: Web search API
- **Similarity Search**: Cosine similarity for vector retrieval

### Development Process

1. **Initial Setup**: Google ADK exploration, agent architecture design
2. **PDF Pipeline**: Implemented LangChain document loading and embedding
3. **Vector Search**: ChromaDB integration with HuggingFace embeddings
4. **LLM Integration**: Connected Gemini 2.0 Flash via LangChain
5. **Web Search**: Added DuckDuckGo search capability
6. **Intelligent Routing**: Implemented LangGraph conditional workflow
7. **UI Development**: Streamlit interface with multi-session support
8. **Optimization**: 
   - Reduced chunk sizes (500 chars) for token efficiency
   - Limited retrieval to top-3 results
   - Single API call per query architecture

### Challenges Overcome

1. **API Quota Issues**: Switched from agent-based to direct LLM calls to reduce API consumption
2. **Model Compatibility**: Navigated Gemini model limitations (function calling support)
3. **Routing Logic**: Refined keyword detection for accurate PDF vs Web routing
4. **Event Loop Management**: Resolved async/threading issues in Streamlit
5. **Package Deprecation**: Migrated from `duckduckgo-search` to `ddgs`

---

## If I Had More Time, This Is What I'd Do

### 1. **Enhanced Multi-Modal Support**
- Extract and analyze images/tables from PDFs
- Support for multiple file formats (DOCX, PPTX, Excel)
- OCR for scanned documents

### 2. **Advanced Retrieval**
- Implement hybrid search (semantic + keyword BM25)
- Re-ranking with cross-encoders for better relevance
- Query expansion and reformulation
- Parent-child document chunking for better context

### 3. **Improved Web Search**
- Multiple search engine aggregation (Google, Bing, DuckDuckGo)
- Web page content extraction and summarization
- Fact-checking and source credibility scoring
- Citation and reference management

### 4. **Collaborative Features**
- Share sessions with team members
- Collaborative annotations on PDFs
- Export conversation history as reports
- Team knowledge base creation

### 5. **Smarter Routing**
- LLM-based intent classification instead of keyword matching
- Learn from user feedback to improve routing
- Multi-source synthesis (combine PDF + Web automatically)
- Follow-up question handling with context memory

### 6. **Performance & Scalability**
- Caching for frequently asked questions
- Batch processing for multiple PDFs
- Background processing for large documents
- Database optimization for faster retrieval

### 7. **User Experience**
- Document preview with highlighted relevant sections
- Interactive source citations (click to view)
- Voice input/output support
- Mobile-responsive design
- Dark mode

### 8. **Enterprise Features**
- User authentication and access control
- Audit logs for compliance
- Private deployment options
- Custom model fine-tuning
- API for programmatic access

### 9. **Quality Improvements**
- Automated testing suite
- Response quality metrics and monitoring
- A/B testing for routing strategies
- User feedback loop for continuous improvement

### 10. **Advanced Agent Capabilities**
- Multi-agent debate for complex questions
- Self-reflection and answer verification
- Tool creation and dynamic API integration
- Proactive information gathering
- Learning from past conversations

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Setup
1. Create `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Run
```bash
streamlit run langgraph_app.py
```

### Usage
1. Create a new session
2. Upload a PDF document (optional)
3. Ask questions - the system automatically routes to PDF or web search
4. Switch between sessions to work with different documents

---

## License
MIT

## Contributors
By Overclocked Minds -> Aarav Aggarwal, Aakriti Vasudev, Anushka Dixit

## Acknowledgments
- Google for Gemini API and ADK framework
- LangChain team for excellent tooling
- HuggingFace for open-source embeddings
