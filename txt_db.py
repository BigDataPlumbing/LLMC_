import os
import openai 
import streamlit as st
import json
import csv
import time
import pinecone
from tqdm.auto import tqdm  # this is our progress bar

#from pinecone_datasets import load_dataset

from streamlit_chat import message

import base64
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

#from datasets import load_dataset

from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt,
    PineconeReader,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI



openai.api_key = st.secrets["openai"]

content_count = 0



MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "The processing pipeline (dataflow) that you define is modeled using a graph.",
        "This graph models the relations between the different connectors,",
        "In that mode, finite and static data can be loaded, e.g. from a table written in a static csv file or from a markdown table."
    ], engine=MODEL
)

def data_to_firestore(message):
        # Use the application default credentials to init firebase
    if not firebase_admin._apps:
        
        uploaded_file = st.file_uploader("Choose a file", type=["json"])

        if uploaded_file is not None:
            # Display the file details
            st.write("File Details:")
            st.write(uploaded_file.name)
            st.write(uploaded_file.type)
            st.write(uploaded_file.size)
            
            # Display the result or perform further actions
            st.write("Certificate Path:", cert_path)

        if uploaded_file is not None:
            cert_path = uploaded_file
            with open(cert_path) as cert:
                project_id = json.load(cert).get('project_id')
            if not project_id:
                raise ValueError('Failed to determine project ID from service account certificate.')
            return credentials.Certificate(cert_path), project_id
    
            db = firestore.client()

            # Writing data to Firebase
            doc_ref = db.collection(u'string').document(message.get('txt_id'))
            doc_ref.set({
                u'User': True,
                u'Assistant': message.get('txt'),
                u'Content': "@",
                u'Note': message.get('nE'),
            })
        


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
content_concluded = False
content_prompted = False
        


if prompt := st.chat_input("Query:"):

    model= "gpt-3.5-turbo-0613",
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = ""
        content_prompted = True
                
        for response in openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-0613",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            result += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(result + "▌")
            content_count += 1
        message_placeholder.markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    
    content_concluded = True
    
    
    
    

if content_concluded and content_prompted and content_count > 1:
    print(f"Responses count: {content_count}")
    
    
urls = [
    "./wrk"
]



def create_pages(urls):

    pages = []
    for url in urls:
        pagename = url.split("/")[-1]
        pages.append(pagename)

    return pages

def build_files(pages):
    docs = {}
    #for page in pages:
    #    docs[page] = SimpleDirectoryReader(
    #        input_files=[f"{page}.txt"]
    #    ).load_data()
    docs = {'In that mode, finite and static data can be loaded, e.g. from a table written in a static json file or from a spreed table.'}
    return docs


def build_folders(model_name):
    OAI_api_key = st.secrets["openai"]
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name, openai_api_key = OAI_api_key)
    )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)



def build_index(pages, docs):

    pine_api_key = st.secrets["pinek"]
    pine_api_env = st.secrets["pines"]
    
    page_indices = {}
    pinecone.init(
        api_key=pine_api_key,
        environment=pine_api_env
    )

    
    pinecone_index = pinecone.Index("data")


    

    service_data = build_folders("gpt-3.5-turbo-0613")

    for page in pages:
        
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            metadata_filters={"page": page}
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        page_indices[page] = GPTVectorStoreIndex.from_documents(
            docs, storage_context=storage_context, service_context=service_data
        )
        page_indices[page].index_struct.index_id = page



    print("Indexing complete.")
    return page_indices
    
    

if __name__ == "__main__":
    
    log_txt = {'@type': 'type.googleapis.com/google.pubsub.v1.PubsubMessage','attributes': {'txt_Id': 'Words', 'NumId': '123', 'RegId': 'logs', 'RegistryLocation': 'nam5','gatewayId': 'logs_gateway', 'projectId': 'emerald-pipe-400817'}, 'data': 'abc123'}
    #context = None
    data_to_firestore(log_txt)
    
    pages = create_pages(urls)
    
    #pages = "wrk"
    docs = build_files(pages)
    # print(docs.keys())
    indices = build_index(pages, docs)
