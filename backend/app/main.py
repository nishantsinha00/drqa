from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated
from pydantic import BaseModel
from typing import List, Union, Optional
from fastapi import Request, Query
import typing as t
import uvicorn

import datetime

import azure.ai.vision as sdk

from scipy.io import wavfile

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool

from langchain.chains.summarize import load_summarize_chain

from langchain.callbacks import get_openai_callback

import textwrap
from dotenv import load_dotenv

import os
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# docs = []
# from sentence_transformers import SentenceTransformer

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
class Reader:
    def __init__(self, img):
        self.img = img
        service_options = sdk.VisionServiceOptions(os.environ["AZURE_VISION_ENDPOINT"],
                                           os.environ["AZURE_VISION_KEY"])
        vision_source = sdk.VisionSource(filename=self.img)
        analysis_options = sdk.ImageAnalysisOptions()
        analysis_options.features = (
            sdk.ImageAnalysisFeature.CAPTION |
            sdk.ImageAnalysisFeature.TEXT
            )
        analysis_options.language = "en"
        analysis_options.gender_neutral_caption = True
        self.analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

    def __call__(self):
        return self.extract_text()

    def extract_text(self):
        result = self.analyzer.analyze()

        lines = []

        for line in result.text.lines:
            lines.append(line.content)
            
        extracted_text = "\n".join(lines)

        return extracted_text
    
def speech_recognize_continuous_from_file(filename):
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=os.environ['AZ_SPEECH_KEY'], region=os.environ['AZ_SPEECH_REGION'])
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    recognized_text = ''  # Initialize an empty string for recognized text

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def recognized_cb(evt):
        """callback that handles the recognized event"""
        nonlocal recognized_text
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text += evt.result.text

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        pass

    speech_recognizer.stop_continuous_recognition()

    return recognized_text
    
def img_to_doc(img_file):
    docs = []
    reader = Reader(img=img_file)
    text = reader()
    docs.append(Document(page_content=text, metadata={"source": img_file, 'page': 0}))
    docs = text_splitter.split_documents(docs)
    for doc in docs:
        doc.metadata['date']= datetime.date.today().strftime('%Y-%m-%d')
        
    return docs

def audio_to_doc(audio_file):
    docs = []
    recognized_text = speech_recognize_continuous_from_file(filename=audio_file)
    sample_rate, data = wavfile.read(audio_file)
    len_data = len(data)
    t = len_data / sample_rate
    docs.append(Document(page_content=recognized_text, metadata={"source": audio_file, 'Audio Length': t}))
    docs = text_splitter.split_documents(docs)
    for doc in docs:
        doc.metadata['date']= datetime.date.today().strftime('%Y-%m-%d')
        
    return docs
          
def pdf_to_doc(pdf_file):
    i = 0
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    docs = text_splitter.split_documents(pages)
    for doc in docs:
        doc.metadata['date']=datetime.date.today().strftime('%Y-%m-%d')
        
    return docs

def get_qa_chain(namespace = None, dateRange = None):
    docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace = namespace)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    prompt_template = """Use the following pieces of context to answer the question at the end. If the information is not
    provided in the context, please do not make one up.
    
    {context}
    
    Question: {question}
    """
    if dateRange is None:
        filter_kwarg = {}
    else:
        filter_kwarg = {"date": {"$in": dateRange}}
        
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo"),
        docsearch.as_retriever(search_kwargs = {"filter": filter_kwarg}),
        memory = memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    ) 
    return qa

def get_summary(docs):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose = True)
    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    return wrapped_text

def create_file_summaries_table(conn):
    try:
        cursor = conn.cursor()
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS file_summaries (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                summary TEXT NOT NULL
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
    except Exception as ex:
        print("Error creating file_summaries table:", str(ex))

def save_file_summary(filename, file_type, summary, conn):
    try:
        cursor = conn.cursor()
        insert_query = "INSERT INTO file_summaries (filename, file_type, summary) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (filename, file_type, summary))
        conn.commit()
        cursor.close()
    except Exception as ex:
        print("Error saving file summary:", str(ex))

def fetch_file_summary_by_filename(filename, conn):
    try:
        cursor = conn.cursor()
        select_query = "SELECT summary FROM file_summaries WHERE filename = %s"
        cursor.execute(select_query, (filename,))
        file_summary = cursor.fetchone()
        conn.commit()
        cursor.close()
        if file_summary:
            return file_summary
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

    create_file_summaries_table(conn)

    for file in files:
        filename = file.filename
        summary = "success"
        print(file.size)
        try:
            if not os.path.exists('app/documents'):
                os.makedirs('app/documents')
            filepath = os.path.join('app','documents', os.path.basename(filename))
            contents = await file.read()
            with open(filepath, 'wb') as f:
                f.write(contents)
            mimestart = mimetypes.guess_type(filepath)[0].split('/')
            if(mimestart[1] == '.pdf'):
                docs.extend(pdf_to_doc(filepath))
            elif(mimestart[0] == 'image'):
                docs.extend(img_to_doc(filepath))
            elif (mimestart[0] == 'audio'):
                docs.extend(audio_to_doc(filepath))
            summary = get_summary(docs=docs)
            save_file_summary(filename, file_type, summary, conn)
        except Exception as ex:
            print(str(ex))
            summary = "error"
            if filepath is not None and os.path.exists(filepath):
                os.remove(filepath)
            # raise HTTPException(summary_code=500, detail="Your file received but couldn't be stored!")
    
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
    conn.close()
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
def get_file_summary_by_filename(filename: str):
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
        file_summary = fetch_file_summary_by_filename(filename, conn)
        return file_summary
    except HTTPException as http_ex:
        # Propagate the HTTPException if raised in the fetch_file_summary_by_filename function
        raise http_ex
    finally:
        # Close the database connection
        conn.close()

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