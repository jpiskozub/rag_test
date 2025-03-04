#%%
#IMPORTS
from pprint import pprint
import json
from pathlib import Path
import nltk
import re
import os



from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient

from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# %%
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"

loader = PyPDFLoader(pdf_url)

pages = loader.load_and_split()

#%%
latex_text = r"""
\documentclass{article}

\begin{document}

\maketitle

\section{Introduction}

Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.

\subsection{History of LLMs}

The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

\subsection{Applications of LLMs}

LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

\end{document}
"""
# %%


for i, page in enumerate(pages):
    print(f"\n--- Page {i+1} ---\n")
    print(page.page_content)

# Section splitter
sections = re.split(r'\\section{(.+?)}', latex_text)
parsed_sections = {}
for i in range(1, len(sections), 2):
    title = sections[i]
    content = sections[i + 1].strip()
    parsed_sections[title] = content


pprint(parsed_sections)



#%%
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://eu-de.ml.cloud.ibm.com",
    project_id="bf575760-1fc7-46cf-9066-5a14330dd45b",
    params=embed_params,
)
# %%
query = "How are you?"

print(watsonx_embedding.embed_query(query))
# %%
