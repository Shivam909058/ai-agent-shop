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
        data_path = os.path.join(os.path.dirname(__file__), "data.txt")
        
        if not os.path.exists(data_path):
            st.error("data.txt file not found! Please ensure it exists with Agno agent code.")
            return None
            
        
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
        prompt = f"""You are an expert Agno/Phidata developer. Based on these Agno agent implementations:

        {context}

        Create a complete, working implementation of a {agent_type} using Agno framework.
        Return only the Python code without any additional text or markdown formatting.
        
        The code should include:
        1. Required imports
        2. Agent configuration with appropriate tools
        3. Specific instructions and descriptions
        4. Example usage code
        """
        
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        response = message.content
        if isinstance(response, list):
            code = "\n".join(block.text for block in response if hasattr(block, 'text'))
        else:
            code = str(response)
        
        code = code.replace("```python", "").replace("```", "").strip()
        return code
        
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
                    generated_code = generate_agent_code(agent_type, context)
                    
                    if generated_code:
                        st.success("Agent code generated successfully!")
                        st.code(generated_code, language="python")
                        
                        st.download_button(
                            label="Download Agent Code",
                            data=generated_code,
                            file_name=f"{agent_type.lower().replace(' ', '_')}_agent.py",
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
