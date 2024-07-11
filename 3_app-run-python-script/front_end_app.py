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

# Define global variables
llm_choice = get_supported_models()
embed_models = get_supported_embed_models()

def update_active_collections(collection_name):
    global collection_list_items
    collection_list_items = get_active_collections()
    print(f"new collection {collection_list_items}")
    collection = ""
    if collection_name is not None and len(collection_name) != 0:
        collection = collection_name
    elif len(collection_list_items) != 0:
        collection = collection_list_items[0]
    return collection

def get_latest_default_collection():
    collection_list_items = get_active_collections()
    collection = ""
    if len(collection_list_items) != 0:
        collection = collection_list_items[0]
    return collection

llm = CMLLLM()
llm.set_collection_name(collection_name=get_latest_default_collection())

def upload_document_and_ingest_new(files, questions, collection_name, progress_bar):
    if files is None or len(files) == 0:
        st.error("Please add some files...")
    return llm.ingest(files, questions, collection_name, progress_bar)

def reconfigure_llm(model_name, embed_model_name, temperature, max_new_tokens, context_window, gpu_layers, progress_bar):
    llm.set_global_settings_common(
        model_name=model_name,
        embed_model_path=embed_model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        n_gpu_layers=gpu_layers,
        progress_bar=progress_bar,
    )
    st.success("Done reconfiguring llm!!!")

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

def open_chat_accordion():
    st.markdown("### Chat with your documents")
    st.session_state['chat_accordion_open'] = True

def close_doc_process_accordion():
    st.session_state['doc_process_accordion_open'] = False

def get_runtime_information():
    if check_gpu_enabled():
        st = "AI chatbot is running using GPU."
    else:
        st = "AI chatbot is running using CPU."
    st += f"\n Using the model {llm.get_active_model_name()}."
    st += f"\n Using the embed model {llm.get_active_embed_model_name()}."
    return st

def demo():
    st.title("AI Chat with your documents")

    # Initialize session state variables
    if 'chat_accordion_open' not in st.session_state:
        st.session_state['chat_accordion_open'] = False
    if 'doc_process_accordion_open' not in st.session_state:
        st.session_state['doc_process_accordion_open'] = True
    if 'collection_name' not in st.session_state:
        st.session_state['collection_name'] = "default_collection"
    if 'nr_of_questions' not in st.session_state:
        st.session_state['nr_of_questions'] = 1

    # Header
    st.markdown("<center><h2>AI Chat with your documents</h2></center>", unsafe_allow_html=True)
    st.markdown("<h3>Chat with your documents (pdf, text and html)</h3>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["Chat with your document", "Advanced Options"])

    with tab1:
        # Document Processing Section
        with st.expander("Process your documents", expanded=st.session_state['doc_process_accordion_open']):
            st.text_area(
                "Now using the collection",
                value=f"AI Chat with your document. Now using the collection {get_latest_default_collection()}",
                max_chars=None,
                key="header",
                height=50,
            )
            uploaded_files = st.file_uploader(
                "Upload your pdf, html or text documents (single or multiple)",
                type=file_types,
                accept_multiple_files=True
            )
            db_progress = st.empty()
            if st.button("Click to process the files"):
                progress_bar = st.progress(0)
                result = upload_document_and_ingest_new(uploaded_files, st.session_state['nr_of_questions'], st.session_state['collection_name'], progress_bar)
                db_progress.text(result)
                open_chat_accordion()

        # Chat Section
        if st.session_state['chat_accordion_open']:
            st.markdown("### Chat with your documents")
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            user_message = st.text_input("Your message:")
            if st.button("Submit"):
                st.session_state['history'].append([user_message, None])
                response = infer2(user_message, st.session_state['history'], st.session_state['collection_name'])
                for message in response:
                    st.session_state['history'][-1][1] = message
                st.session_state['history'].append(["", None])
            for user_msg, bot_msg in st.session_state['history']:
                st.write(f"**User:** {user_msg}")
                st.write(f"**Bot:** {bot_msg}")

    with tab2:
        # Advanced Options Section
        st.text_area(
            "LLM processing status",
            value=get_runtime_information(),
            height=50,
        )
        st.slider(
            "Number of questions to be generated per document",
            0, 10, 1,
            key="nr_of_questions",
            on_change=lambda: st.session_state.update(nr_of_questions=st.session_state['nr_of_questions'])
        )
        with st.expander("Collection configuration"):
            collection_list = st.selectbox(
                "Configure an existing collection or create a new one below",
                get_active_collections(),
                key="collection_list"
            )
            if st.button("Refresh the collection list"):
                st.session_state['collection_list'] = get_active_collections()
            if st.button("Delete the collection and the associated document embeddings"):
                llm.delete_collection_name(st.session_state['collection_list'], st.progress(0))
                st.session_state['collection_list'] = get_active_collections()
            collection_name = st.session_state['collection_list']
            llm.set_collection_name(collection_name)

if __name__ == "__main__":
    demo()
