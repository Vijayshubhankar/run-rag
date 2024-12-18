import os
import asyncio
import pg8000
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from google.cloud.sql.connector import Connector
from langchain_community.vectorstores.pgvector import PGVector
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import storage

def download_pdf_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a PDF file from Google Cloud Storage."""
    # Initialize a GCS client
    storage_client = storage.Client()

    # Get the bucket and blob (file) from GCS
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file to the local system
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from GCS to {destination_file_name}")

download_pdf_from_gcs("scienct_text_book", "10th-english-science.pdf", "data.pdf")

# Path to the science text book.
pdf_path = "data.pdf"

# Function to extract text from the science text book.
async def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

pdf_pages = asyncio.run(load_pdf(pdf_path))

# print(f"{pdf_pages[0].metadata}\n")
# print(pdf_pages[0].page_content)

# Split the pages into chunks of 1000 characters
def split_pages(pdf_pages):
    texts = []
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=0
        )
    for page in pdf_pages:
        text = text_splitter.split_text(page.page_content)
        texts.append(text[0])
    return texts

text_chunks = split_pages(pdf_pages)

# Set up a PGVector instance 
connector = Connector()

def getconn() -> pg8000.dbapi.Connection:
    """Function to connect to the Cloud SQL PostgreSQL database."""
    conn: pg8000.dbapi.Connection = connector.connect(
        os.getenv("DB_INSTANCE_NAME", ""),
        "pg8000",
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", ""),
    )
    return conn

# Initialize PGVector with VertexAI embeddings
store = PGVector(
    connection_string="postgresql+pg8000://",
    use_jsonb=True,
    engine_args=dict(
        creator=getconn,
    ),
    embedding_function=VertexAIEmbeddings(
        model_name="textembedding-gecko@003"
    ),
    pre_delete_collection=True  
)

# Save all text chunks in the Cloud SQL database
ids = store.add_texts(text_chunks)

print(f"Done saving: {len(ids)} text chunks")