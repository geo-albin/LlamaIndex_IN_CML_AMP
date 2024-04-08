import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import UnstructuredReader, PDFReader
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from huggingface_hub import hf_hub_download
import time
import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from utils.duplicate_preprocessing import DuplicateRemoverNodePostprocessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from utils.upload import Upload_files
import torch
import logging
import sys
import subprocess
import gradio as gr
import atexit
import utils.vectordb as vectordb
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# from transformers import BitsAndBytesConfig

index_created = False


def exit_handler():
    print("cmlllmapp is exiting!")
    vectordb.stop_vector_db()


atexit.register(exit_handler)

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])


MODELS_PATH = "./models"
EMBED_PATH = "./embed_models"

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,
    local_files_only=True,
)

embed_model = "thenlper/gte-large"

n_gpu_layers = 20

Settings.llm = LlamaCPP(
    model_path=model_path,
    temperature=0.0,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={"temperature": 0.0, "top_k": 5, "top_p": 0.95},
    # generate_kwargs={"temperature": 0.0},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": n_gpu_layers},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=embed_model,
    cache_folder=EMBED_PATH,
)

Settings.callback_manager = callback_manager

node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=20)
# Settings.node_parser = node_parser

# node_parser = SentenceWindowNodeParser.from_defaults(
#     window_size=3,
#     window_metadata_key="window",
#     original_text_metadata_key="original_text",
# )

Settings.node_parser = node_parser
Settings.text_splitter = SentenceSplitter()

print(subprocess.run(["rm -f questions.txt"], shell=True))

milvus_start = vectordb.reset_vector_db()
print(f"milvus_start = {milvus_start}")
collection = vectordb.create_vector_db_collection()
print(f"milvus collection created = {collection}")

vector_store = MilvusVectorStore(
    dim=1024,
    overwrite=True,
    collection_name="cml_rag_collection",
)

# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

postprocessor = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    threshold_cutoff=0.7,
)


def Infer(query, history=None):
    print(f"query = {query}")

    query_text = ""
    if isinstance(query, dict) and query["text"]:
        query_text = query["text"]
    elif isinstance(query, str):
        query_text = query
    else:
        yield ""
        return

    if len(query_text) == 0:
        yield "Please ask some questions"
        return

    global index
    global index_created

    if index_created == False:
        yield "Please process some document in step 1."
        return

    global chat_engine
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        postprocessor=[
            # MetadataReplacementPostProcessor(target_metadata_key="window"),
            postprocessor,
            DuplicateRemoverNodePostprocessor(),
        ],
    )

    streaming_response = chat_engine.stream_chat(query_text)
    generated_text = ""
    for token in streaming_response.response_gen:
        generated_text = generated_text + token
        yield generated_text


def list_files():
    file_paths = []
    directory = os.path.abspath("./assets/doc_list")
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def Ingest(questions, progress=gr.Progress()):
    file_extractor = {
        ".html": UnstructuredReader(),
        ".pdf": PDFNougatOCR(),
        # ".pdf": PDFReader(),
        ".txt": UnstructuredReader(),
    }

    print(f"questions = {questions}")

    progress(0.3, desc="loading the documents")

    # documents = []

    try:
        start_time = time.time()
        files = list_files()
        for file in files:
            # reader = SimpleDirectoryReader(
            #     input_dir="./assets/doc_list",
            #     recursive=True,
            #     file_extractor=file_extractor,
            # )
            progress(0.4, desc=f"loading document {file}")
            reader = SimpleDirectoryReader(
                input_files=[file],
                file_extractor=file_extractor,
            )
            document = reader.load_data(num_workers=1, show_progress=True)

            progress(0.4, desc="done loading document {file}")

            # vector_store = MilvusVectorStore(
            #     dim=1024,
            #     # overwrite=True,
            #     collection_name="cml_rag_collection",
            # )

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # progress(
            #     0.45, desc="done starting the vector db and set the storage context..."
            # )

            progress(0.4, desc=f"start indexing the document {file}")
            # index = VectorStoreIndex.from_documents(
            #     documents=documents, storage_context=storage_context, show_progress=True
            # )
            nodes = node_parser.get_nodes_from_documents(document)

            global index
            index = VectorStoreIndex(
                nodes, storage_context=storage_context, show_progress=True
            )
            progress(0.4, desc=f"done indexing the document {file}")
            # documents.append(document)
            # while len(documents) > 10:
            #     documents.pop(0)

            op = (
                "Completed data ingestion. took "
                + str(time.time() - start_time)
                + " seconds."
            )

            print(f"{op}")
            progress(0.4, desc=op)

            start_time = time.time()
            print(f"start dataset generation from the document {file}.")
            progress(0.4, desc=f"start dataset generation from the document {file}.")
            data_generator = DatasetGenerator.from_documents(documents=document)

            dataset_op = (
                f"Completed data set generation for file {file}. took "
                + str(time.time() - start_time)
                + " seconds."
            )
            op += "\n" + dataset_op
            print(f"{dataset_op}")
            progress(0.4, desc=dataset_op)
            print(f"start generating questions from the document {file}")
            progress(0.4, desc=f"start generating questions from the document {file}")
            eval_questions = data_generator.generate_questions_from_nodes(num=questions)

            i = 1
            for q in eval_questions:
                op += "\nQuestion " + str(i) + " - " + str(q)
                i += 1

            write_list_to_file(eval_questions, "questions.txt")
            print(f"done generating questions from the document {file}")
            progress(0.4, desc=f"done generating questions from the document {file}")

        progress(0.9, desc="done processing the documents...")
        print("done processing the documents...")
        global index_created
        index_created = True
    except Exception as e:
        print(e)
        op = f"ingestion failed with exception {e}"
        progress(0.9, desc=op)

    return op


def write_list_to_file(lst, filename):
    with open(filename, "a") as f:
        for item in lst:
            f.write(str(item) + "\n")


def upload_document_and_ingest(files, questions, progress=gr.Progress()):
    if len(files) == 0:
        return "Please add some files..."
    Upload_files(files, progress)
    return Ingest(questions, progress)


def clear_chat_engine():
    if index_created == True:
        chat_engine.reset()


if __name__ == "__main__":
    Ingest(True)
