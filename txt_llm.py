import os
import openai 
import streamlit as st

from streamlit_chat import message


openai.api_key = st.secrets["openai"]


st.set_page_config(
    page_title="LLM db",
    page_icon="",
    menu_items={
         'Get Help': 'https://www.web.com/help',
         'Report a bug': "https://www.web.com/bug",
         'About': "# This is a page."
     }
)



content_count = 0
w_count = 0
content_w = 0



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    if message["role"] != "prompt":
        if message["role"] != "data":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
content_concluded = False
content_prompted = False


    

if "content_store" not in st.session_state:
    st.session_state.content_store = []

for message in st.session_state.content_store:
    content_w = 0



def add_prompt_content(responseq):
        st.session_state.content_store.append({"role": "prompt", "content": responseq})
        
def add_data_content(responsed):
        st.session_state.content_store.append({"role": "data", "content": responsed})
        
        
st.sidebar.write(" ")        

    
    
    
    
if prompt := st.chat_input("Text?"):

    add_prompt_content(prompt)
    
    
    
    model= "gpt-3.5-turbo-0613",
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        

        

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        content_prompted = True
                
        for response in openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-0613",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
            content_count += 1
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    content_concluded = True
    
    
    
    

if content_concluded and content_prompted and content_count > 1 and w_count == 0:

    

    w_response = ""
    
    for response in openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": "Write about next content: " + prompt}
            ],
            stream=True,
        ):
            w_response += response.choices[0].delta.get("content", "")




if content_count > 1:

    with st.sidebar:
        st.write("Writting ...")
    w_count += 1
    






  

if content_count > 1: 


    add_data_content(w_response)
    

if content_count > 1: 

        
    with st.sidebar:
        st.write(" ")

        
        for spec_item in st.session_state.content_store:
            if spec_item["role"] == "data":
                st.write(spec_item["content"])
            #    st.write(data_item["content"])
            if content_count > 1: 
                if spec_item["role"] == "prompt":
                    st.write(" ")
                    st.write("~")
                    st.write(spec_item["content"])
            else:
                if spec_item["role"] == "prompt":
                    st.write(spec_item["content"])
        st.write("~")
        st.write(" ")
