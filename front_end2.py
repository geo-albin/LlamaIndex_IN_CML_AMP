import os
import streamlit as st
from utils.cmlllm import (
    CMLLLM,
    get_active_collections,
    get_supported_embed_models,
    get_supported_models,
    infer2,
)
from utils.check_dependency import check_gpu_enabled

MAX_QUESTIONS = 5

file_types = ["pdf", "html", "txt"]

llm_choice = get_supported_models()
embed_models = get_supported_embed_models()
progress = st.progress

def save_uploadedfile(uploadedfile):
    save_path = os.path.join("uploaded_files", uploadedfile.name)
    with open(save_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return save_path

def reconfigure_llm(
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    embed_model_name="thenlper/gte-large",
    temperature=0.0,
    max_new_tokens=1024,
    context_window=3900,
    gpu_layers=20,
):
    llm.set_global_settings_common(
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
    collection = ""
    if len(collection_list_items) != 0:
        collection = collection_list_items[0]

    return collection

def validate_llm(model_name, embed_model_name):
    ret = True
    if model_name is None or len(model_name) == 0:
        st.error("Select a valid model name")
        ret = False

    if embed_model_name is None or len(embed_model_name) == 0:
        st.error("Select a valid embed model name")
        ret = False

    return ret

def validate_collection_name(collectionname):
    ret = True
    if collectionname is None or len(collectionname) == 0:
        st.error("Invalid collection name, please set a valid collection name string.")
        ret = False

    return ret

def upload_document_and_ingest_new(
    files, questions, collection_name, progress
):
    if files is None or len(files) == 0:
        print("upload files")
    output = st.session_state.llm.ingest(files, questions, collection_name, st.progress(0))
    return output

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

header = get_latest_default_collection()

def demo():
    st.title("AI Chat with your documents")
    files = []
    collection_list_items = ["default_collection"]
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF/HTML/TXT Files", type=file_types, accept_multiple_files=True)
        collection_name = st.selectbox(
            "Select Collection",
            st.session_state.collection_list_items
        )
        saved_files = []
        if st.session_state.get('processing', False):
            submit_button_disabled = True
        else:
            submit_button_disabled = False
        if st.button("Submit & Process", disabled=submit_button_disabled):
            if uploaded_files:
                st.session_state['processing'] = True
                for uploaded_file in uploaded_files:
                    st.spinner("@@@@@@@@@@...")
                    st.spinner(uploaded_file)
                    saved_path = save_uploadedfile(uploaded_file)
                    saved_files.append(saved_path)

            if uploaded_files:
                with st.spinner("Processing..."):
                    questions = upload_document_and_ingest_new(saved_files, st.session_state.num_questions, collection_name, progress)
                    st.success("Done")
                    st.session_state['documents_processed'] = True
                    st.session_state['questions'] = questions
                    st.session_state['processing'] = False
                    st.session_state.used_collections.append(collection_name)  # Track used collection
                    if questions:
                        st.text_area("Auto-Generated Questions", questions)

        st.checkbox("Advanced Settings", value=False, key='advanced_settings')

        if st.session_state['advanced_settings']:
            num_questions = st.slider("Number of question generations", min_value=1, max_value=10, value=st.session_state.num_questions, key='num_questions')
            if num_questions != st.session_state.num_questions:
                st.session_state.num_questions = num_questions
            with st.expander("Collection configuration"):
                custom_input = st.text_input("Enter your custom collection name:")
                if st.button("Refresh the collection list") and custom_input:
                    st.session_state.collection_list_items.append(custom_input)
                    st.experimental_rerun()

    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Link and ask me anything about the content.'}]
    if 'documents_processed' not in st.session_state:
        st.session_state['documents_processed'] = False
    if 'questions' not in st.session_state:
        st.session_state['questions'] = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
    if st.session_state['documents_processed']:
        user_prompt = st.chat_input("Ask me anything about the content of the document:")
        if uploaded_files:
            st.session_state.messages = [{'role': 'assistant', "content": f'Using collection: {collection_name}'}]

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.write(user_prompt)

            with st.spinner("Thinking..."):
                response = infer2(user_prompt, "", collection_name)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            with st.chat_message("assistant"):
                st.write(response)
    else:
        st.write(
            "Documents are not yet processed. Please upload and process documents before asking questions."
        )

if __name__ == "__main__":
    demo()
