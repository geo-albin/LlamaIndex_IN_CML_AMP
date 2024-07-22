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
from streamlit_option_menu import option_menu

MAX_QUESTIONS = 5

file_types = ["pdf", "html", "txt"]

llm_choice = get_supported_models()
collection_list_items = get_active_collections()
embed_models = get_supported_embed_models()

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

def upload_document_and_ingest_new(files, questions, collection_name, progress_bar):
    if files is None or len(files) == 0:
        print("upload files")
    output = llm.ingest(files, questions, collection_name, progress_bar)
    return output

llm = CMLLLM()
llm.set_collection_name(collection_name=collection_list_items[0])

def demo():
    """
    Main function to run the Streamlit app.
    """
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Link and ask me anything about the content.'}]

    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            "Main Menu", ["Home", "Chat App", "Setting"],
            icons=['cloud-upload', 'chat', 'gear'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "5!important"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "grey"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    # Home Tab: File Upload and Processing
    if selected == "Home":
        st.title("AI Chat with your documents")
        uploaded_files = st.file_uploader("Upload your PDF/HTML/TXT Files", type=file_types, accept_multiple_files=True, key="home_upload")
        if st.button("Submit & Process"):
            if uploaded_files:
                saved_files = [save_uploadedfile(file) for file in uploaded_files]
                st.session_state.uploaded_files.extend(saved_files)
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    questions = upload_document_and_ingest_new(saved_files, st.session_state.get('nr_of_questions', 1), st.session_state.get('collection_name', "default_collection"), progress_bar)
                    st.success("Done")
                    st.session_state['documents_processed'] = True
                    st.session_state['questions'] = questions
                    st.session_state['processing'] = False
                    if questions:
                        st.text_area("Auto-Generated Questions", questions)

    # Display uploaded files in the sidebar
    if st.session_state.uploaded_files:
        st.sidebar.subheader("Uploaded Files:")
        for file_path in st.session_state.uploaded_files:
            st.sidebar.write(os.path.basename(file_path))

    # Chat App Tab: Chat with Uploaded Documents
    if selected == "Chat App":
        st.subheader("Chat with your document")
        if 'documents_processed' not in st.session_state:
            st.session_state['documents_processed'] = False
        if 'questions' not in st.session_state:
            st.session_state['questions'] = []

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
                    response = next(infer2(user_prompt, "", st.session_state.get('collection_name', "default_collection")))
                st.session_state.messages.append({'role': 'assistant', "content": response})

                with st.chat_message("assistant"):
                    st.write(response)
        else:
            st.write("Documents are not yet processed. Please upload and process documents before asking questions.")

    # Setting Tab: Configuration Options
    if selected == "Setting":
        st.title("Setting")
        st.session_state.nr_of_questions = st.slider("Number of questions to be generated per document", 0, 10, 1)
        st.session_state.collection_name = st.selectbox(
            "Select Collection",
            options=get_active_collections(),
            index=0
        )

        st.write("Selected Collection:", st.session_state.collection_name)
        if st.button("Refresh Collections"):
            st.experimental_rerun()
            st.success("Collection Refreshed")

        if st.button("Delete Collection"):
            if st.session_state.collection_name:
                llm.delete_collection_name(st.session_state.collection_name)
                st.success(f"Collection {st.session_state.collection_name} deleted.")
            else:
                st.error("Please select a collection to delete.")

if __name__ == "__main__":
    demo()
