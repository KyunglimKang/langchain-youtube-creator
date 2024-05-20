# Import os to set API key
import os
from apikey import apikey
# Bring in streamlit for UI/app interface
import streamlit as st 
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Pinecone
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 


def pdf():

    # App framework
    st.title('🦜🔗 PDF')
    prompt = st.text_input('Plug in your prompt here')     

    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.1, verbose=True)
    # Create and load PDF Loader
    loader = PyPDFLoader('./docs/annualreport.pdf')

    # Split pages from pdf 
    pages = loader.load_and_split()

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'], 
        template='write me a youtube video title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'], 
        template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    )

    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    wiki = WikipediaAPIWrapper()
    
    # Show stuff to the screen if there's a prompt
    if prompt: 
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt) 
        script = script_chain.run(title=title, wikipedia_research=wiki_research)

        st.write(title) 
        st.write(script) 

        with st.expander('Title History'): 
            st.info(title_memory.buffer)

        with st.expander('Script History'): 
            st.info(script_memory.buffer)

        with st.expander('Wikipedia Research'): 
            st.info(wiki_research)