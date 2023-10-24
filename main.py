import streamlit as st
# Install Modules
# !pip install transformers==2.8.0
# !pip install torch==1.4.0
# !pip install datsets transformers[sentencepiece]
# !pip install sentencepiece

st.set_page_config(layout="wide")

# Import Module
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from PyPDF2 import PdfReader

# initialize the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')



@st.cache_resource
def text_summary(text):

     ## preprocess the input text
     preprocessed_text = text.strip().replace('\n', '')
     t5_input_text = 'summarize: ' + preprocessed_text
     tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
     summary_ids = model.generate(tokenized_text, min_length=50, max_length=120)
     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

     return summary

def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text


choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using txtai")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document using txtai")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
