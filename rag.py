#%%
#IMPORTS
from pprint import pprint

import re
import os
import gradio as gr




from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient

from langchain_community.document_loaders import TextLoader
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# %%
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"

loader = PyPDFLoader(pdf_url)

pages = loader.load_and_split()

print(pages[0].page_content[:1000])

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

#%%

loader = TextLoader("new-Policies.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(data)

# %%
ids = [str(i) for i in range(0, len(chunks))]
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

query = "Smoking policy"
docs = vectordb.similarity_search(query, k=5)
print(docs)

#%%
query = "Email policy"

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)
print(docs)

#%%
model_id = 'mistralai/mixtral-8x7b-instruct-v01'
llm_parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://eu-de.ml.cloud.ibm.com",
    project_id="bf575760-1fc7-46cf-9066-5a14330dd45b",
    params=llm_parameters,
)

#%%


## QA Chain
def retriever_qa(file, query):
    llm = watsonx_llm
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

#%%
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)
# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port= 7860)