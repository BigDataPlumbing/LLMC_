import os
import openai 
import streamlit as st
import json
import csv
import time
import tempfile
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

#from llama_index import (
#    SimpleDirectoryReader,
#    LLMPredictor,
#    ServiceContext,
#    GPTVectorStoreIndex,
#    QuestionAnswerPrompt,
#    PineconeReader,
#)

from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI



openai.api_key = st.secrets["openai"]

content_count = 0



MODEL = "text-embedding-ada-002"

#res = openai.Embedding.create(
#    input=[
#        "The processing pipeline (dataflow) that you define is modeled using a graph.",
#        "This graph models the relations between the different connectors,",
#        "In that mode, finite and static data can be loaded, e.g. from a table written in a static csv file or from a markdown table."
#    ], engine=MODEL
#)


def data_to_firestore(message):
        # Use the application default credentials to init firebase
    if not firebase_admin._apps:
        
        uploaded_file = st.file_uploader("Choose a file", type=["json"])
        credentials_obj = None
        project_id_returned = None
        app = None

        if uploaded_file is not None:
            # Display the file details
            st.write("File Details:")
            st.write(uploaded_file.name)
            st.write(uploaded_file.type)
            st.write(uploaded_file.size)

                    # Get the content of the uploaded file
            file_content = uploaded_file.read()

        # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_content)
                cert_path = temp_file.name

                    # Display the certificate path
            st.write(f"Certificate Path: {cert_path}")

        app = 1   

        if uploaded_file is not None:
            st.write("Project obj:")
            #cert_path = uploaded_file
            with open(cert_path) as cert:
                project_id = json.load(cert).get('project_id')
                st.write(project_id)
                credentials_obj = credentials.Certificate(cert_path)
                st.write("Certificate:", credentials_obj)
                project_id_returned = project_id
            if not project_id:
                st.write("Failed to determine project ID from service account certificate.")
                raise ValueError('Failed to determine project ID from service account certificate.')
            #return credentials.Certificate(cert_path), project_id
            return
        
        st.write(" - - - ")
        #st.write(app)
            
        if app is not None:
            

                    # Display the result or perform further actions
            st.write("Certificate app:", cert_path)
            st.write("Project:", project_id_returned)

            default_app = firebase_admin.initialize_app(credentials_obj, {'projectId': 'emerald-pipe-400817', })
    
            db = firestore.client()

            # Writing data to Firebase
            doc_ref = db.collection(u'string').document(message.get('txt_id'))
            doc_ref.set({
                u'User': True,
                u'Assistant': message.get('txt'),
                u'Content': "@",
                u'Note': message.get('nE'),
            })
            


    
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
    docs['wrk'] = {'In that mode, finite and static data can be loaded, e.g. from a table written in a static json file or from a spreed table.'}
    return docs


def build_folders(model_name):
    #OAI_api_key = st.secrets["openai"]
    #llm_predictor = LLMPredictor(
    #    llm=ChatOpenAI(temperature=0, model_name=model_name, openai_api_key = OAI_api_key)
    #)
    return #ServiceContext.from_defaults(llm_predictor=llm_predictor)



def build_index():

    # def build_index(pages, docs):

    pine_api_key = st.secrets["pinek"]
    pine_api_env = st.secrets["pines"]
    
    page_indices = {}
    pinecone.init(
        api_key=pine_api_key,
        environment=pine_api_env
    )

    
    pinecone_index = pinecone.Index("data")


    pinecone.create_index("data", dimension=1536, metric="cosine")
    # pinecone_index = pinecone.Index("data")
    # pinecone_index.upsert("data", [1,2,3])
    # pinecone_index.describe_index_stats()

    service_data = build_folders("gpt-3.5-turbo-0613")

    #for page in pages:
        
        #vector_store = PineconeVectorStore(
        #    pinecone_index=pinecone_index,
        #    metadata_filters={"page": page}
        #)
        #storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #page_indices = GPTVectorStoreIndex.from_documents(
        #    docs, storage_context=storage_context, service_context=service_data
        #)
        #page_indices.index_struct.index_id = page



    print("Indexing complete.")
    return 



        


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
    #build_index()
    log_txt = {'@type': 'type.googleapis.com/google.pubsub.v1.PubsubMessage','attributes': {'txt_Id': 'Words', 'NumId': '123', 'RegId': 'logs', 'RegistryLocation': 'nam5','gatewayId': 'logs_gateway', 'projectId': 'emerald-pipe-400817'}, 'data': 'abc123'}
    data_to_firestore(log_txt)

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

    
    

if __name__ == "__main__":
    
    log_txt = {'@type': 'type.googleapis.com/google.pubsub.v1.PubsubMessage','attributes': {'txt_Id': 'Words', 'NumId': '123', 'RegId': 'logs', 'RegistryLocation': 'nam5','gatewayId': 'logs_gateway', 'projectId': 'emerald-pipe-400817'}, 'data': 'abc123'}
    data_to_firestore(log_txt)
    st.write("Details:")
    
    pages = create_pages(urls)
    
    docs = build_files(pages)
    
