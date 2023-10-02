PDF reader documentation

Llm model used is: orca-mini-3b.ggmlv3.q4_0.bin
The orca-mini-3b.ggmlv3.q4_0.bin model is a variant of the Orca Mini 3B model trained by Pankaj Mathur and it's based on the OpenLLaMa-3B model. It has been trained on "explain tuned" datasets, which were created using instructions and input from the WizardLM, Alpaca, and Dolly-V2 datasets.

# PDF Reader and Question Answering

This Python script performs PDF text extraction and question answering using various libraries and models. It is designed to help users extract information from PDF documents and generate responses to their questions.

## Features

- Extract text from PDF documents using `PyPDF2` and `pdfplumber`.
- Split extracted text into paragraphs using a custom text splitter.
- Initialize a sentence embedding model for sentence similarity.
- Encode each paragraph and select the most relevant ones.
- Generate responses to user questions using a GPT-4 language model.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- [PyPDF2](https://pythonhosted.org/PyPDF2/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [SentenceTransformer](https://github.com/UKPLab/sentence-transformers)
- [GPT4All](https://github.com/chatGPT/GPT4All)
- [Numba](https://numba.pydata.org/)

## Usage

1. Clone this Git repository to your local machine.

2. Install the required Python libraries in a new environment:
    pip install -r requirements.txt

3. specify the path of pdf in the line 102 of pdf_query_answer_generator.py

    python pdf_query_answer_generator.py

4. You can go through the pdf_query_answer_generator.ipynb file for more details

Video demo of CLI is:
https://www.youtube.com/watch?v=j51714XPRjo
