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
import shutil

MAX_QUESTIONS = 5
file_types = ["pdf", "html", "txt"]
llm_choice = get_supported_models()
embed_models = get_supported_embed_models()
lock = threading.Lock()

def save_uploadedfile(uploadedfile, collection_name):
    save_dir = os.path.join("uploaded_files", collection_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, uploadedfile.name)
    with open(save_path, "wb") as f:
        f.write(uploadedfile.getbuffer())

    return save_path

def delete_collection_name(collection_name):
    # Remove the corresponding folder
    collection_dir = os.path.join("uploaded_files", collection_name)
    if os.path.exists(collection_dir):
      shutil.rmtree(collection_dir)

def list_files_in_collection(collection_name):
    collection_dir = os.path.join("uploaded_files", collection_name)
    filelist = []
    if os.path.exists(collection_dir):
        files = os.listdir(collection_dir)
        for f in files:
          filelist.append(os.path.join(collection_dir, f))
        return filelist
    return []

def get_collection_folders(directory="uploaded_files"):
    try:
        # Get all folders in the specified directory
        collection_folders = [folder for folder in os.listdir(directory)
                              if os.path.isdir(os.path.join(directory, folder))]
        return collection_folders
    except FileNotFoundError:
        return []

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

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_uploadedfile, file, collection_name) for file in files]
        saved_files = [future.result() for future in futures]
    collection_files = list_files_in_collection(collection_name)
    if collection_files == []:
      return "No files"

    output = st.session_state.llm.ingest(collection_files, questions, collection_name)
    return output

# Initialize session state if not already present
if 'llm' not in st.session_state:
    with st.spinner('Initializing LLM...'):
        st.session_state.llm = CMLLLM()
if 'collection_list_items' not in st.session_state:
    exiting_collection = get_collection_folders()
    all_collection = list(set(["default_collection"] + exiting_collection))
    st.session_state.collection_list_items = all_collection
    st.session_state.llm.set_collection_name(collection_name=st.session_state.collection_list_items[0])
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 1
if 'used_collections' not in st.session_state:
    st.session_state.used_collections = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": f'Hello! You are using {st.session_state.collection_list_items[0]} folder.'}]
if 'documents_processed' not in st.session_state:
    st.session_state['documents_processed'] = False
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'success_message' not in st.session_state:
    st.session_state['success_message'] = ""
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = st.session_state.collection_list_items[0]

header = get_latest_default_collection()

def refresh_session_state_on_collection_change(collection_name):
    st.session_state.llm.set_collection_name(collection_name=collection_name)
    st.session_state.current_collection = collection_name
    st.session_state.messages = [{'role': 'assistant', "content": f'Hello! You are using {collection_name} folder.'}]
    st.session_state.documents_processed = False
    st.session_state.questions = []
    st.session_state.processing = False
    st.session_state.success_message = ""

def demo():
    st.title("AI Chat with Your Documents")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF/HTML/TXT Files", type=file_types, accept_multiple_files=True)
        collection_name = st.selectbox(
            "Select Folder",
            st.session_state.collection_list_items
        )
        c = st.expander(f"Existing files in : {collection_name}")
        if collection_name != st.session_state.get('current_collection'):
            refresh_session_state_on_collection_change(collection_name)
            # st.session_state.llm.set_collection_name(collection_name=collection_name)
            # st.session_state.current_collection = collection_name
            # # Update the initial message with the new collection
            st.session_state.messages[0]['content'] = f'Hello! You are using {collection_name} folder.'
        items = None
        if st.session_state.get('current_collection'):
          dir_path = os.path.join("uploaded_files", collection_name)
          if os.path.exists(dir_path):
            items = os.listdir(dir_path)
            if items:
              for item in items:
                c.write(item)
            else:
              c.write("No docs found")
          else:
            c.write(f"{collection_name} is empty")
        if st.button("Submit & Process", disabled=st.session_state.processing):
            if uploaded_files or items:
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
                    st.text_area("Auto-Generated Questions", st.session_state['questions'], key='auto_generated_questions')
        st.write("")  # Add empty line for space
        st.write("")  # Add another empty line for more space
        st.checkbox("Advanced Settings", value=st.session_state.get('advanced_settings', False), key='advanced_settings')

        if st.session_state['advanced_settings']:
            num_questions = st.slider("Number of question generations", min_value=0, max_value=MAX_QUESTIONS, value=st.session_state.num_questions, key='num_questions')
            if num_questions != st.session_state.num_questions:
                st.session_state.num_questions = num_questions
            with st.expander("Folder Configuration"):
                custom_input = st.text_input("Enter your custom folder name:")
                if st.button("Add to the folder list") and custom_input:
                    custom_input = custom_input.rstrip().replace(" ", "_")
                    if custom_input not in st.session_state.collection_list_items:
                      st.session_state.collection_list_items.append(custom_input)
                      st.session_state['success_message'] = f"Folder {custom_input} added"
                      st.experimental_rerun()
                    else:
                      st.warning(f'Folder {custom_input} already exists, try other name')
                if st.button("Delete the Selected Folder") and collection_name != "default_collection":
                    st.session_state.collection_list_items.remove(collection_name)
                    st.session_state.llm.delete_collection_name(collection_name)
                    st.session_state['success_message'] = f"Folder {collection_name} deleted"
                    delete_collection_name(collection_name)
                    st.experimental_rerun()
                elif collection_name == "default_collection":
                    st.error("You can't delete the default_collection")

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
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.write(user_prompt)

            with st.spinner("Thinking..."):
              response = infer2(user_prompt, "", st.session_state.current_collection)
              response1, response2 = itertools.tee(response)
              with st.chat_message("assistant"):
                st.write_stream(response1)
              complete_response = "".join(list(response2))
              st.session_state.messages.append({"role": "assistant", "content": complete_response})
    else:
        st.write("Documents are not yet processed. Please upload and process documents before asking questions.")

if __name__ == "__main__":
    demo()
