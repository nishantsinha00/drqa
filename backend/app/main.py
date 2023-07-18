from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated
from pydantic import BaseModel
from typing import List, Union
from fastapi import Request, Query
import typing as t
import uvicorn

import datetime

import azure.ai.vision as sdk

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

load_dotenv()
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
docs = []
# from sentence_transformers import SentenceTransformer


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
def img_to_doc(img_file):
    docs = []
    reader = Reader(img=img_file)
    text = reader()
    docs.append(Document(page_content=text, metadata={"source": img_file, 'page': 0}))
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
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo"),
        docsearch.as_retriever(search_kwargs = {"filter": {"date": {"$in": dateRange}}}),
        memory = memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    ) 
    return qa

class UserQuery(BaseModel):
    query: str




@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}



@app.post("/upload-files")
async def upload_file(request: Request, 
                      files: List[UploadFile],
                      file_type: str = Query("Lab Report", enum=["Lab Report", "Prescription", "Other"]),
                      ):
    
    for file in files:
        filename = file.filename
        status = "success"
        print(file.size)
        try:
            if not os.path.exists('app/documents'):
                os.makedirs('app/documents')
            filepath = os.path.join('app','documents', os.path.basename(filename))
            contents = await file.read()
            with open(filepath, 'wb') as f:
                f.write(contents)
            if(os.path.splitext(filepath)[1] == '.pdf'):
                docs.extend(pdf_to_doc(filepath))
            else:
                docs.extend(img_to_doc(filepath))
        except Exception as ex:
            print(str(ex))
            status = "error"
            if filepath is not None and os.path.exists(filepath):
                os.remove(filepath)
            # raise HTTPException(status_code=500, detail="Your file received but couldn't be stored!")
    
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=file_type)
    return {"filename": filename, "status": status}

@app.post("/")



@app.post("/query")
async def query_index(request: Request, 
                      input_query: UserQuery, 
                      namespace: str = Query("Lab Report", enum=["Lab Report", "Prescription", "Other"]),
                      dateRange: Union[List, None] = None):
    print(dateRange)
    qa_chain = get_qa_chain(namespace = namespace, dateRange = dateRange)
    result = qa_chain({"question": input_query.query})
    print(result)
    return {"response": result['answer'], "relevant_docs": result['source_documents']}


@app.post("/summarize")
async def summarize(request: Request):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose = True)
    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    return {"response": wrapped_text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)