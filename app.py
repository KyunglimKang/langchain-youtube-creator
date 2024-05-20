# Bring in deps
import os 
from apikey import apikey 
from youtube import youtube
from pdf import pdf
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# Create the sidebar menu
st.sidebar.title("Menu")
menu_selection = st.sidebar.radio("Go to", ["Home", "youtube", "PDF"])

# Define page content
if menu_selection == "Home":
    st.write("Welcome to the Home page!")
elif menu_selection == "youtube":
    youtube()
elif menu_selection == "PDF":
    st.write("This is Page 1.")
    pdf()
else:
    st.write("You are on Page 2.")
