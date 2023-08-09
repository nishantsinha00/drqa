import openai

from ExtractionProperties import *

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, create_extraction_chain_pydantic
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS


from langchain.chains.summarize import load_summarize_chain

import textwrap

from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io import wavfile

import azure.ai.vision as sdk

import datetime

import json
import os

from dotenv import load_dotenv

load_dotenv()

index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

with open('promts.json', 'rb') as f:
    prompts = json.load(f)

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

def preproc_audio(file_path):
    basePath, audio_format = os.path.splitext(file_path)
    audio_format = audio_format[1:]
    sound = AudioSegment.from_file(file_path)
    audio_chunks = split_on_silence(sound
                            ,min_silence_len = 1000
                            ,silence_thresh = -45
                            ,keep_silence = 200
                        )
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    wavfilepath = basePath + '.wav'
    combined.export(wavfilepath, format='wav')
    return wavfilepath

def audio_to_doc(audio_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    i = 0
    docs = []
    
    new_audio_path = preproc_audio(audio_path)
    sample_rate, data = wavfile.read(new_audio_path)
    name = os.path.basename(new_audio_path)
    len_data = len(data)
    t = len_data / sample_rate
    audio_file= open(new_audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=os.environ["OPENAI_API_KEY"])
    recognized_text = transcript['text']
    docs.append(Document(page_content=recognized_text, metadata={"source": name, 'Audio Length': t}))
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
    provided in the context, please do not make one up, if the information  provided in the context, has incorrect spelling of the diseases, medications, and labs make sure that you correct it and replace it with the help of the internet. 
    Assume that you are a care advocate and are helping the patient and their caregivers better keep track of patient-related visits, documents, and other needs.
    Keep your responses concise, but make sure to include all important information, names of diseases, medications, and labs. 
    Respond in an easy-to-understand, empathetic tone. 
    
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


def get_doc_data(docs, fileType):
    llm = OpenAI(temperature=0)
    if fileType == "End of Visit Summaries":
        map_prompt = """
             Write a concise summary of the following:
             "{text}"
             CONCISE SUMMARY:
             """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt = """
        Write a concise summary of the following text delimited by triple backquotes.
        Summarize this physician visit as if you are trying to explain what you heard.
        ```{text}```
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        llm = OpenAI(temperature=0)
        summarization_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     )
    else:  
        summarization_chain = load_summarize_chain(llm, 
                                     chain_type="map_reduce",
                                     verbose = True)
        
    output_summary = summarization_chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    #vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    prompt_template = """Use the following pieces of context to answer the question at the end. If the information is not
    provided in the context, please do not make one up, if the information  provided in the context, has incorrect spelling of the diseases, medications, and labs make sure that you correct it and replace it with the help of the internet. 
    Assume that you are a care advocate and are helping the patient and their caregivers better keep track of patient-related visits, documents, and other needs.
    Make sure to include all important information, names of diseases, medications, and labs. 
    Respond in an easy-to-understand, empathetic tone as if you are explaining to the patient. 
    
    {context}
    
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=1, model="gpt-3.5-turbo-0613"),
        vectorstore.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    ) 
    questions = prompts[fileType]
    text = ""
    for question in questions:
        result = qa({"question": question})
        text += result["answer"] + "\n"

    print(text)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    
    if fileType == "Prescriptions":
        chain = create_extraction_chain_pydantic(pydantic_schema=PrescriptionProperties, llm=llm)
    elif fileType == "Insurance documents":
        chain = create_extraction_chain_pydantic(pydantic_schema=InsuranceDocumentProperties, llm=llm)
    elif fileType == "Discharge notes":
        chain = create_extraction_chain_pydantic(pydantic_schema=DischargeSummaryProperties, llm=llm)
    elif fileType == "End of Visit Summaries":
        chain = create_extraction_chain_pydantic(pydantic_schema=VisitSummaries, llm=llm)
    else:
        chain = create_extraction_chain_pydantic(pydantic_schema=GeneralProperties, llm=llm)

    extracted_data = chain.run(text)
    dataDict = extracted_data[0].dict()
    dataDict['summary'] = wrapped_text

    return dataDict