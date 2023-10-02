from PyPDF2 import PdfReader
from pdfplumber import pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import time
from gpt4all import GPT4All

from numba import jit, cuda
import numpy as np  


import sys,time,random

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

# Loading the llm model here
llm = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

def get_query():
    query = input("Enter your question\n")
    progressBar(1, 7)
    return query


def load_split_pdf(pdf_path):
    pdf_loader = PdfReader(open(pdf_path, "rb"))
    pdf_text = ""
    for page_num in range(len(pdf_loader.pages)):
        pdf_page = pdf_loader.pages[page_num]
        pdf_text += pdf_page.extract_text()
    progressBar(2, 7)
    return pdf_text


def split_text_using_RCTS(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=64
    )
    split_texts = text_splitter.split_text(pdf_text)
    paragraphs = []
    for text in split_texts:
        paragraphs.extend(text.split('\n')) 
    progressBar(3, 7)
    return paragraphs


def Initialize_sentence_transformer():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformer(model_name)
    progressBar(4, 7)
    return embeddings


def encode_each_paragraph(paragraphs, embeddings):
    responses = []
    for paragraph in paragraphs:
        response = embeddings.encode([paragraph], convert_to_tensor=True)
        responses.append((paragraph, response))
    progressBar(5, 7)
    return responses


def choose_most_relevant_sentence(embeddings, responses, query):
    query_embedding = embeddings.encode([query], convert_to_tensor=True)
    best_response = None
    best_similarity = -1.0
    answers = []

    for paragraph, response in responses:
        
        similarity = util.pytorch_cos_sim(query_embedding, response).item()
        
        if similarity >= 0.6:
            
            # count += 1
            
            answers.append(paragraph)
    answer = "\n".join(answers)
    progressBar(6, 7)
    return answer


def query_the_llm(answer, llm_model, query):
    prompt_message = answer + "\n" + query

    final_response = llm_model.generate(prompt=prompt_message)
    
    return final_response

    

def main(llm):
    start_time = time.time()
    
    pdf_path = "D:/project/pdf_reader/pdf_reader/pdf_reader_v2/document/AltoK10_Owner's_Manual.pdf"
    
    query = get_query()
    
    pdf_text = load_split_pdf(pdf_path)
    
    paragraphs = split_text_using_RCTS(pdf_text)
    
    embeddings = Initialize_sentence_transformer()
    
    responses = encode_each_paragraph(paragraphs=paragraphs, embeddings=embeddings)
    
    answer = choose_most_relevant_sentence(embeddings=embeddings, responses=responses, query=query)
    
    final_response = query_the_llm(answer=answer, llm_model=llm, query=query)
    
    print ("The answer from model is\n", final_response)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Execution time: {elapsed_time/60} minutes \n")
    
    progressBar(7, 7)

main(llm)