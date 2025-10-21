import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import PyPDF2 # Import PyPDF2 for PDF processing
import io # Import io for handling file-like objects

# Preprocessing function (simplified as TfidfVectorizer handles tokenization and stopwords)
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def calculate_similarity(doc1_content, doc2_content):
    if not doc1_content or not doc2_content:
        return 0.0

    # Preprocess documents
    processed_doc1 = preprocess_text(doc1_content)
    processed_doc2 = preprocess_text(doc2_content)

    if not processed_doc1 or not processed_doc2:
        return 0.0

    # Create TF-IDF vectorizer with built-in English stop words
    vectorizer = TfidfVectorizer(stop_words=\'english\') # Corrected: \'english\' instead of \\\\'english\\\\'
    tfidf_matrix = vectorizer.fit_transform([processed_doc1, processed_doc2])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

# Function to read text from PDF
def read_pdf_content(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() or ""
    return text

# Function to read content based on file type
def read_file_content(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf_content(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        return None

st.title("Document Similarity Checker")
st.write("Upload two documents (.txt or .pdf) to find their similarity score.")

# File uploaders
uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf"])
uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf"])

doc1_content = None
doc2_content = None

if uploaded_file1 is not None:
    doc1_content = read_file_content(uploaded_file1)
    if doc1_content:
        st.success("Document 1 uploaded successfully!")

if uploaded_file2 is not None:
    doc2_content = read_file_content(uploaded_file2)
    if doc2_content:
        st.success("Document 2 uploaded successfully!")

if st.button("Calculate Similarity"):
    if doc1_content and doc2_content:
        similarity = calculate_similarity(doc1_content, doc2_content)
        st.subheader("Similarity Score:")
        st.write(f"The similarity between the two documents is: {similarity:.2f}")
        st.progress(similarity)
    else:
        st.warning("Please upload both documents to calculate similarity.")


