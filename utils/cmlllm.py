import os
import asyncio
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.readers.file import UnstructuredReader, PDFReader
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from huggingface_hub import hf_hub_download, snapshot_download
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
import torch
import logging
import sys
import subprocess
import streamlit as st
import atexit
import utils.vectordb as vectordb
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from utils.common import supported_llm_models, supported_embed_models

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

QUESTIONS_FOLDER = "questions"

def exit_handler():
    print("cmlllmapp is exiting!")
    vectordb.stop_vector_db()

atexit.register(exit_handler)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])

def get_supported_embed_models():
    embedList = list(supported_embed_models)
    return embedList

chat_engine_map = {}

def get_supported_models():
    llmList = list(supported_llm_models)
    return llmList

active_collection_available = {"default_collection": False}

def get_active_collections():
    return list(active_collection_available)

print("resetting the questions")
print(subprocess.run([f"rm -rf {QUESTIONS_FOLDER}"], shell=True))

milvus_start = vectordb.reset_vector_db()
print(f"milvus_start = {milvus_start}")


def infer2(msg, history, collection_name):
    query_text = msg
    print(f"query = {query_text}, collection name = {collection_name}")

    if len(query_text) == 0:
        return "Please ask some questions"

    if collection_name in active_collection_available and not active_collection_available[collection_name]:
        return "No documents are processed yet. Please process some documents.."

    if collection_name not in chat_engine_map:
        return f"Chat engine not created for collection {collection_name}.."

    chat_engine = chat_engine_map[collection_name]

    try:
        streaming_response = chat_engine.stream_chat(query_text)
        generated_text = ""
        for token in streaming_response.response_gen:
            yield token
#            generated_text += token
#        return generated_text
    except Exception as e:
        op = f"failed with exception {e}"
        print(op)
        return op

class CMLLLM:
    MODELS_PATH = "./models"
    EMBED_PATH = "./embed_models"
    questions_folder = QUESTIONS_FOLDER

    def __init__(
        self,
        model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        embed_model_name="thenlper/gte-large",
        temperature=0.0,
        max_new_tokens=1024,
        context_window=3900,
        gpu_layers=20,
        dim=1024,
        collection_name="default_collection",
        memory_token_limit=3900,
        sentense_embedding_percentile_cutoff=0.8,
        similarity_top_k=2,
        progress_bar=None,  # Add progress_bar parameter
    ):
        if len(model_name) == 0:
            model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        if len(embed_model_name) == 0:
            embed_model_name = "thenlper/gte-large"
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_name
        n_gpu_layers = 0
        if torch.cuda.is_available():
            print("It is a GPU node, setup GPU.")
            n_gpu_layers = gpu_layers

        self.node_parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=128)

        self.set_global_settings(
            model_name=model_name,
            embed_model_path=embed_model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            node_parser=self.node_parser,
            progress_bar=progress_bar,
        )
        self.dim = dim
        self.similarity_top_k = similarity_top_k
        self.sentense_embedding_percentile_cutoff = sentense_embedding_percentile_cutoff
        self.memory_token_limit = memory_token_limit

    def get_active_model_name(self):
        print(f"active model is {self.active_model_name}")
        return self.active_model_name

    def get_active_embed_model_name(self):
        print(f"active embed model is {self.active_embed_model_name}")
        return self.active_embed_model_name

    def delete_collection_name(self, collection_name, progress_bar=None):
        print(f"delete_collection_name : collection = {collection_name}")

        if collection_name is None or len(collection_name) == 0:
            return None

        active_collection_available.pop(collection_name, None)
        chat_engine_map.pop(collection_name, None)

    def set_collection_name(
        self,
        collection_name,
        progress_bar=None,
    ):
        print(f"set_collection_name : collection = {collection_name}")

        if collection_name is None or len(collection_name) == 0:
            return None

        print(f"adding new collection name {collection_name}")

        if not collection_name in active_collection_available:
            active_collection_available[collection_name] = False

        if collection_name in chat_engine_map:
            print(
                f"collection {collection_name} is already configured and chat_engine is set"
            )
            return

        vector_store = MilvusVectorStore(
            dim=self.dim,
            collection_name=collection_name,
        )

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


        chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            postprocessor=[
                SentenceEmbeddingOptimizer(
                    percentile_cutoff=self.sentense_embedding_percentile_cutoff
                ),
                DuplicateRemoverNodePostprocessor(),
            ],
            memory=ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit),
            system_prompt=(
                "You are an expert Q&A assistant that is trusted around the world.\n"
                "Always answer the query using the Context provided and not prior knowledge or General knowledge."
                "Avoid statements like 'Based on the context' or 'The context information'.\n"
                "If the provided context dont have the information, answer 'I dont know'.\n"
                "Please cite the source along with your answers."
            ),
            similarity_top_k=self.similarity_top_k,
        )
        chat_engine_map[collection_name] = chat_engine
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    def ingest(self, files, questions, collection_name, progress_bar=None):
        if not (collection_name in active_collection_available):
            return f"Some issues with the llm and collection {collection_name} setup. please try setting up the llm and the vector db again."

        file_extractor = {
            ".html": UnstructuredReader(),
            ".pdf": PDFReader(),
            ".txt": UnstructuredReader(),
        }

#        if torch.cuda.is_available():
#            file_extractor[".pdf"] = PDFNougatOCR()

        print(f"collection = {collection_name}, questions = {questions}")


        filename_fn = lambda filename: {"file_name": os.path.basename(filename)}

        active_collection_available[collection_name] = False

        try:
            start_time = time.time()
            op = ""
            i = 1
            for file in files:

                reader = SimpleDirectoryReader(
                    input_files=[file], file_extractor=file_extractor, file_metadata=filename_fn
                )
                document = reader.load_data(num_workers=1)

                print(f"document = {document}")

                vector_store = MilvusVectorStore(
                    dim=self.dim,
                    collection_name=collection_name,
                )

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                nodes = self.node_parser.get_nodes_from_documents(document)
                index = VectorStoreIndex(
                    nodes, storage_context=storage_context
                )
                data_generator = DatasetGenerator.from_documents(documents=document)
                dataset_op = (
                    f"Completed data set generation for file {os.path.basename(file)}. took "
                    + str(time.time() - start_time)
                    + " seconds."
                )
                eval_questions = data_generator.generate_questions_from_nodes(
                    num=questions
                )

                for q in eval_questions:
                    op += str(q) + "\n"
                    i += 1
                active_collection_available[collection_name] = True
                i += 1
            print(op)
            return op
        except Exception as e:
            print(f"Exception in ingest: {e}")
            active_collection_available[collection_name] = False
            return f"Error: {e}"

    def upload_document_and_ingest(self, files, questions, progress_bar=None):
        if len(files) == 0:
            return "Please add some files..."
        return self.ingest(files, questions, progress_bar)

    def set_global_settings(
        self,
        model_name,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        node_parser,
        progress_bar=None,
    ):
        self.set_global_settings_common(
            model_name=model_name,
            embed_model_path=embed_model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            n_gpu_layers=n_gpu_layers,
            progress_bar=progress_bar,
        )
        Settings.node_parser = node_parser

    def set_global_settings_common(
        self,
        model_name,
        embed_model_path,
        temperature,
        max_new_tokens,
        context_window,
        n_gpu_layers,
        progress_bar=None,
    ):
        print(
            f"Enter set_global_settings_common. model_name = {model_name}, embed_model_path = {embed_model_path}"
        )
        self.active_model_name = model_name
        self.active_embed_model_name = embed_model_path
        model_path = self.get_model_path(model_name)
        print(f"model_path = {model_path}")

        Settings.llm = LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            generate_kwargs={"temperature": temperature},
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )


        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_path,
            cache_folder=self.EMBED_PATH,
        )


    def get_model_path(self, model_name):
        filename = supported_llm_models[model_name]
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            resume_download=True,
            cache_dir=self.MODELS_PATH,
            local_files_only=True,
            token=hf_token,
        )
        return model_path

    def get_embed_model_path(self, embed_model):
        embed_model_path = snapshot_download(
            repo_id=embed_model,
            resume_download=True,
            cache_dir=self.EMBED_PATH,
            local_files_only=True,
            token=hf_token,
        )
        return embed_model_path

    def clear_chat_engine(self, collection_name):
        if collection_name in chat_engine_map:
            chat_engine = chat_engine_map[collection_name]
            chat_engine.reset()
