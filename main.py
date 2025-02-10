import streamlit as st
from anthropic import Anthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel, Field
from textwrap import dedent
load_dotenv()
os.makedirs("./data/vectordb", exist_ok=True)
anthropic = Anthropic()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n## ", "\n### ", "\n```python", "\n```", "\n", " ", ""],
    keep_separator=True,
    add_start_index=True,
)

def init_embeddings():
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None

embeddings = init_embeddings()

def clean_markdown_content(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('```python'):
            in_code_block = True
            cleaned_lines.append(line)
        elif line.startswith('```') and in_code_block:
            in_code_block = False
            cleaned_lines.append(line)
        elif in_code_block:
            cleaned_lines.append(line)
        elif line.strip() and not line.startswith('|'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
def init_knowledge_base():
    try:
    
        if os.path.exists("./data/vectordb"):
            try:
                
                vectordb = Chroma(
                    persist_directory="./data/vectordb",
                    embedding_function=init_embeddings(),
                    collection_name="agno_agents"
                )
                if vectordb._collection.count() > 0:
                    st.success("Successfully loaded existing knowledge base!")
                    return vectordb
            except Exception as e:
                st.error(f"Error loading existing vector store: {str(e)}")
                return None
        

        data_path = os.path.join(os.path.dirname(__file__), "data.txt")
        
        if not os.path.exists(data_path):
            st.error("No existing knowledge base found and data.txt is missing! Please provide data.txt for first-time initialization.")
            return None
            
        st.info("Initializing new knowledge base from data.txt...")
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = clean_markdown_content(content)
        texts = text_splitter.create_documents([cleaned_content])
        
        if not texts:
            st.error("No content found in data.txt")
            return None
            
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=init_embeddings(),
            persist_directory="./data/vectordb",
            collection_name="agno_agents"
        )
        
        vectordb.persist()
        return vectordb
        
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {str(e)}")
        return None

def search_knowledge_base(query: str, vectordb, top_k: int = 5):
    
    try:
        search_query = f"""Find Agno agent example and implementation for: {query}
        Focus on:
        - Python code blocks
        - Agent configuration
        - Tools and settings
        - Example usage"""

        docs = vectordb.similarity_search(
            search_query,
            k=top_k
        )
        
        relevant_content = []
        for doc in docs:
            content = doc.page_content
            if '```python' in content:
                
                sections = content.split('```')
                for i, section in enumerate(sections):
                    if section.startswith('python'):
                        code = section[6:].strip()  
                        relevant_content.append(code)
            elif 'Agent(' in content or 'from agno' in content:
                
                relevant_content.append(content)
        
        return '\n\n'.join(relevant_content) if relevant_content else None
        
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None

def generate_agent_code(agent_type: str, context: str):
    try:
        code_prompt = f"""You are an expert Agno/Phidata developer. Based on these Agno agent implementations:

        {context}

        Create a complete, working implementation of a {agent_type} using Agno framework.
        The code should be clean, well-documented, and include:
        1. Clear docstrings explaining the purpose and usage
        2. Helpful inline comments

        
        Return ONLY the Python code without any escape characters or string formatting.
        The code should be properly formatted and indented.
        """
        
        info_prompt = f"""Create a clear explanation for the {agent_type} implementation that includes:

        1. Brief Overview:
           - What the agent does
           - Main purpose and use cases

        2. Key Features:
           - List the main capabilities

        Format the response in clean markdown with clear sections and bullet points.
        Make it easy to read and understand, similar to professional documentation.
        """

        
        code_message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.5,
            messages=[{"role": "user", "content": code_prompt}]
        )
        
        
        info_message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.5,
            messages=[{"role": "user", "content": info_prompt}]
        )
        
        
        code_content = str(code_message.content)
        
        
        if "```python" in code_content:
            code = code_content.split("```python")[1].split("```")[0].strip()
        else:
            code = code_content.strip()
        
        
        code = (code
            .replace('\\n', '\n')  
            .replace('\\t', '\t')  
            .replace('\\"', '"')   
            .strip('"')          
            .strip("'")          
            .strip()              
        )
        
        
        info = str(info_message.content)
        if "TextBlock" in info:
            info = info.split("text='")[1].split("'")[0].replace("\\n", "\n")
        
        
        info = info.replace('\\n', '\n').replace('\\t', '\t')
        
        return {
            "code": code,
            "info": info
        }
        
    except Exception as e:
        st.error(f"Code generation failed: {str(e)}")
        return None


st.title("🤖 Advanced AI Agent Generator USING AGNO")
st.write("Generate specialized AI agents using Agno framework with RAG-powered intelligence")


if 'vectordb' not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.vectordb = init_knowledge_base()

agent_type = st.text_input(
    "What type of AI agent would you like to create?",
    placeholder="e.g., travel agent, coding assistant, research agent"
)

with st.expander("Advanced Options"):
    top_k = st.slider("Number of reference examples", 3, 10, 5)
    include_comments = st.checkbox("Include detailed comments", value=True)
    include_examples = st.checkbox("Include usage examples", value=True)

if st.button("Generate Agent"):
    if agent_type:
        with st.spinner("Searching knowledge base..."):
            context = search_knowledge_base(
                query=f"agent implementation for {agent_type}",
                vectordb=st.session_state.vectordb,
                top_k=top_k
            )
            
            if context:
                st.success("Found relevant examples and documentation!")
                
                with st.spinner("Generating specialized agent code..."):
                    generated_result = generate_agent_code(agent_type, context)
                    
                    if generated_result:
                        st.success("Agent code and documentation generated successfully!")
                        
                        
                        st.markdown("## 📖 Agent Documentation")
                        st.markdown(generated_result["info"])
                        
                        
                        st.markdown("## 💻 Generated Code")
                        try:
                            
                            formatted_code = generated_result["code"].strip()
                            st.code(formatted_code, language="python")
                        except Exception as e:
                            st.error(f"Error formatting code: {str(e)}")
                        
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Code",
                                generated_result["code"],
                                file_name=f"{agent_type.lower().replace(' ', '_')}_agent.py",
                                mime="text/plain"
                            )
                        with col2:
                            st.download_button(
                                "📄 Download Documentation",
                                generated_result["info"],
                                file_name=f"{agent_type.lower().replace(' ', '_')}_docs.md",
                                mime="text/plain"
                            )
            else:
                st.error("Could not find relevant documentation. Please try a different agent type.")
    else:
        st.warning("Please enter an agent type!")

with st.sidebar:
    st.header("About")
    st.write("""
    This advanced tool create custom AI agents using the Agno framework.
    
    Features:
    - knowledge base of Agno documentation
    - Intelligent context retrieval
    - Specialized agent generation
    - Best practice implementation
    - Production-ready code
    
    The tool maintains an up-to-date understanding of:
    - Multi-modal agents
    - Knowledge stores with Agentic RAG
    - Tool integration
    - Memory management
    - Structured outputs
    """)
