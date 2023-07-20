# Imports
import os 
from langchain.llms import OpenAI,HuggingFaceHub,GPT4All
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
api_token="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" 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
df = pd.read_csv('mall_customers_data.csv')
llm= HuggingFaceHub(
        repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        model_kwargs={
            "max_new_tokens":1000,
            "top_k": 10,
            "temperature": 1,
            "top_p": 0.5,
        }
    )
#llm = ChatOpenAI(model_name = model_id, temperature=0)
llm=llm#HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"temperature":0.7, })
df = pd.read_csv('mall_customers_data.csv')
agent = create_pandas_dataframe_agent(llm, df, verbose=True, max_iterations=6)
# Setup streamlit app
# Display the page title and the text box for the user to ask the question
st.title('✨ Query your Data ')
prompt = st.text_input("Enter your question to query your PDF documents")
if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response =  agent.run(prompt)


    # Write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

  