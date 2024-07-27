import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from utils.cmlllm import (
    CMLLLM,
    get_active_collections,
    get_supported_embed_models,
    get_supported_models,
    infer2,
)
from utils.check_dependency import check_gpu_enabled
import threading
import itertools

MAX_QUESTIONS = 5
file_types = ["pdf", "html", "txt"]
llm_choice = get_supported_models()
embed_models = get_supported_embed_models()
lock = threading.Lock()

def save_uploadedfile(uploadedfile):
    save_dir = "uploaded_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, uploadedfile.name)
    with open(save_path, "wb") as f:
        f.write(uploadedfile.getbuffer())

    return save_path

def reconfigure_llm(model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", embed_model_name="thenlper/gte-large",
                    temperature=0.0, max_new_tokens=1024, context_window=3900, gpu_layers=20):
    st.session_state.llm.set_global_settings_common(
        model_name=model_name,
        embed_model_path=embed_model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        n_gpu_layers=gpu_layers,
    )
    return "Done reconfiguring llm!!!"

def get_latest_default_collection():
    collection_list_items = get_active_collections()
    if collection_list_items:
        return collection_list_items[0]
    return ""

def validate_llm(model_name, embed_model_name):
    if not model_name:
        st.error("Select a valid model name")
        return False
    if not embed_model_name:
        st.error("Select a valid embed model name")
        return False
    return True

def validate_collection_name(collection_name):
    if not collection_name:
        st.error("Invalid collection name, please set a valid collection name string.")
        return False
    return True

def upload_document_and_ingest_new(files, questions, collection_name):
    if not files:
        return "No files to upload"

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_uploadedfile, file) for file in files]
        saved_files = [future.result() for future in futures]

    output = st.session_state.llm.ingest(saved_files, questions, collection_name)
    return output

# Initialize session state if not already present
if 'llm' not in st.session_state:
    with st.spinner('Initializing LLM...'):
        st.session_state.llm = CMLLLM()
if 'collection_list_items' not in st.session_state:
    st.session_state.collection_list_items = ["default_collection"]
    st.session_state.llm.set_collection_name(collection_name=st.session_state.collection_list_items[0])
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 1
if 'used_collections' not in st.session_state:
    st.session_state.used_collections = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Link and ask me anything about the content.'}]
if 'documents_processed' not in st.session_state:
    st.session_state['documents_processed'] = False
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'success_message' not in st.session_state:
    st.session_state['success_message'] = ""
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = ""

header = get_latest_default_collection()

def demo():
    st.title("AI Chat with Your Documents")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF/HTML/TXT Files", type=file_types, accept_multiple_files=True)
        collection_name = st.selectbox(
            "Select Collection",
            st.session_state.collection_list_items
        )

        if collection_name != st.session_state.get('current_collection'):
            st.session_state.llm.set_collection_name(collection_name=collection_name)
            st.session_state.current_collection = collection_name

        if st.button("Submit & Process", disabled=st.session_state.processing):
            if uploaded_files:
                st.session_state['advanced_settings'] = False
                st.session_state.processing = True
                with st.spinner("Processing..."):
                    with lock:
                        questions = upload_document_and_ingest_new(uploaded_files, st.session_state.num_questions, collection_name)
                    st.success("Done")
                    st.session_state['documents_processed'] = True
                    st.session_state['questions'] = questions
                    st.session_state['processing'] = False
                    st.session_state.used_collections.append(collection_name)
                    st.text_area("Auto-Generated Questions", questions, key='auto_generated_questions')

        st.checkbox("Advanced Settings", value=st.session_state.get('advanced_settings', False), key='advanced_settings')

        if st.session_state['advanced_settings']:
            num_questions = st.slider("Number of question generations", min_value=1, max_value=MAX_QUESTIONS, value=st.session_state.num_questions, key='num_questions')
            if num_questions != st.session_state.num_questions:
                st.session_state.num_questions = num_questions
            with st.expander("Collection Configuration"):
                custom_input = st.text_input("Enter your custom collection name:")
                if st.button("Add to the collection list") and custom_input:
                    st.session_state.collection_list_items.append(custom_input)
                    st.session_state['success_message'] = f"Collection {custom_input} added"
                    st.experimental_rerun()
                if st.button("Delete the Selected Collection") and collection_name != "default_collection":
                    st.session_state.collection_list_items.remove(collection_name)
                    st.session_state.llm.delete_collection_name(collection_name)
                    st.session_state['success_message'] = f"Collection {collection_name} deleted"
                    st.experimental_rerun()
                elif collection_name == "default_collection":
                    st.error("You can't delete the default collection")

                # Display success message if there is one
                if st.session_state['success_message']:
                    st.success(st.session_state['success_message'])
                    st.session_state['success_message'] = ""

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    if st.session_state['documents_processed']:
        user_prompt = st.chat_input("Ask me anything about the content of the document:")
        st.session_state.messages = [{'role': 'assistant', "content": f'Using collection: {collection_name}'}]
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.write(user_prompt)

            with st.spinner("Thinking..."):
                response = infer2(user_prompt, "", st.session_state.current_collection)
                response1, response2 = itertools.tee(response)
                complete_response = ""
                for response_chunk in response2:
                    complete_response += response_chunk

            st.session_state.messages.append({"role": "assistant", "content": complete_response})
            with st.chat_message("assistant"):
                st.write_stream(response1)
    else:
        st.write("Documents are not yet processed. Please upload and process documents before asking questions.")

if __name__ == "__main__":
    demo()
