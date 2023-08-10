from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated
from pydantic import BaseModel
from typing import List, Optional
from fastapi import Request, Query
import uvicorn

from utils import *

from langchain.vectorstores import Pinecone
from langchain.callbacks import get_openai_callback
import pinecone


import textwrap
from dotenv import load_dotenv

import os
import shutil
import mimetypes

import psycopg2

load_dotenv()

DATABASE_HOST = "localhost"
DATABASE_PORT = 5432
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "115211"
DATABASE_NAME = "postgres"

index_name = "langchain-demo"

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], 
              environment=os.environ["PINECONE_ENV"])
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

document_types = [
    "Lab reports",
    "Imaging reports",
    "Genetic reports",
    "Histopathology reports",
    "Prescriptions",
    "Pill bottle photos",
    "Clinician notes",
    "Discharge notes",
    "End of Visit Summaries",
    "Healthcare bills",
    "Legal documents",
    "Insurance documents",
    "Advance care planning",
    "Financial documents",
    "Miscellaneous"
]


app = FastAPI(
    title="DrQA backend API", docs_url="/docs"
)


origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_file_data_table(conn):
    try:
        cursor = conn.cursor()
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS filedata (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                data JSONB NOT NULL
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
    except Exception as ex:
        print("Error creating table:", str(ex))

def save_file_data(filename, file_type, data, conn):
    try:
        data = json.dumps(data)
        cursor = conn.cursor()
        insert_query = f"INSERT INTO filedata (filename, file_type, data) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (filename, file_type, data))
        conn.commit()
        cursor.close()
    except Exception as ex:
        print("Error saving file summary:", str(ex))

def fetch_file_data_by_filename(filename, conn):
    try:
        cursor = conn.cursor()
        select_query = "SELECT data FROM filedata WHERE filename = %s"
        cursor.execute(select_query, (filename,))
        data = cursor.fetchone()
        conn.commit()
        cursor.close()
        if data:
            return data
        else:
            raise HTTPException(status_code=404, detail="File summary not found.")
    except HTTPException:
        raise
    except Exception as ex:
        print("Error fetching file summary:", str(ex))
        raise HTTPException(status_code=500, detail="Error fetching file summary.")

class UserQuery(BaseModel):
    query: str

class DateRange(BaseModel):
    dateRange: Optional[List] = None
    
    class Config:
        schema_extra = {
            "example": {
            }
        }

@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}


@app.post("/upload-files")
async def upload_file(request: Request, 
                      files: List[UploadFile],
                      file_type: str = Query(document_types[0], enum=document_types),
                      ):
    conn = psycopg2.connect(
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
        dbname=DATABASE_NAME
    )
    docs = []
    create_file_data_table(conn)

    for file in files:
        filename = file.filename
        summary = "success"
        try:
            if not os.path.exists('documents'):
                os.makedirs('documents')
            filepath = os.path.join('documents', os.path.basename(filename))
            contents = await file.read()
            with open(filepath, 'wb') as f:
                f.write(contents)
            mimestart = mimetypes.guess_type(filepath)[0].split('/')
            if(mimestart[1] == 'pdf'):
                with get_openai_callback() as callback:
                    docs.extend(pdf_to_doc(filepath))
                    data = get_doc_data(docs, file_type)
                    print(callback)
            elif(mimestart[0] == 'image'):
                with get_openai_callback() as callback:
                    docs.extend(img_to_doc(filepath))
                    data = get_doc_data(docs, file_type)
                    print(callback)
            elif (mimestart[0] == 'audio'):
                with get_openai_callback() as callback:
                    docs.extend(audio_to_doc(filepath))
                    data = get_doc_data(docs, file_type)
                    print(callback)
            save_file_data(filename, file_type, data, conn)
        except Exception as ex:
            print(str(ex))
            summary = "error"
            if filepath is not None and os.path.exists(filepath):
                os.remove(filepath)
            # raise HTTPException(summary_code=500, detail="Your file received but couldn't be stored!")
    conn.close()

    if os.path.exists('documents'):
        shutil.rmtree('documents')
        
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=file_type)
    return {"filename": filename, "summary": summary}

@app.post("/query")
async def query_index(request: Request, 
                      input_query: UserQuery, 
                      dateRange: DateRange,
                      namespace: str = Query(document_types[0], enum=document_types),
                      ):
    dateRange = dateRange.dateRange
    qa_chain = get_qa_chain(namespace = namespace, dateRange = dateRange)
    result = qa_chain({"question": input_query.query})
    return {"response": result['answer'], "relevant_docs": result['source_documents']}

@app.post("/file-summary")
def get_file_data_by_filename(filename: str):
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        user=DATABASE_USER,
        password=DATABASE_PASSWORD,
        database=DATABASE_NAME
    )

    try:
        # Fetch file summary from the database based on the filename
        file_data = fetch_file_data_by_filename(filename, conn)
        return file_data
    except HTTPException as http_ex:
        # Propagate the HTTPException if raised in the fetch_file_summary_by_filename function
        raise http_ex
    finally:
        # Close the database connection
        conn.close()
@app.post("/report-data")
def get_report_data():
    # Connect to the PostgreSQL database
    with open('ReportData.json', 'rb') as f:
        data = json.load(f)
    return data

"""@app.post("/summarize")
async def summarize(request: Request):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose = True)
    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    return {"response": wrapped_text}"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)