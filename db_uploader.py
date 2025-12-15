from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from os import listdir
from os.path import isfile, join

import db_setup

MAX_KEYWORDS = 10

def trim_keywords(meta: dict) -> dict:
    kw = meta.get("keywords")

    if isinstance(kw, str):
        parts = [k.strip() for k in kw.split(", ") if k.strip()]
        meta["keywords"] = ", ".join(parts[:MAX_KEYWORDS])

    return meta

files_directory = "./data"

vector_store = db_setup.get_db()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=600,
    length_function=len,
    is_separator_regex=False,
)

for file in listdir(files_directory):
    file_path = join(files_directory, file)
    if not isfile(file_path):
        continue

    print(f"Document {file} processing started.")

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Document {file} loaded.")

    texts = text_splitter.split_documents(docs)
    print(f"Document {file} splitted.")

    ids = []
    documents = []

    chunk_id = 1
    for text in texts:
        text.metadata["source"] = file
        text.metadata = trim_keywords(text.metadata)
        ids.append(f"{file}-page{text.metadata["page"] + 1}-chunk{chunk_id}")
        documents.append(text)

        chunk_id += 1

    print(f"Document {file} chunks processed.")

    vector_store.add_documents(documents=documents, ids=ids)

    print(f"Document {file} chunks saved.")
