import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# UI layout
st.title("Resume Ranker AI")
st.write("Paste a job description and upload multiple resumes to rank them by relevance.")

# Input for job description
job_text = st.text_area("Paste Job Description Here", height=200)

# Upload resumes
uploaded_files = st.file_uploader(
    "Upload Resume .txt Files (Multiple Allowed)", type=["txt"], accept_multiple_files=True
)

# Button to trigger ranking
if st.button("Rank Resumes"):

    # Validation
    if not job_text:
        st.warning("Please paste a job description before ranking.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        with st.spinner("Ranking resumes..."):
            # Embed job description
            job_emb = model.encode(job_text)

            resume_scores = []
            for file in uploaded_files:
                resume_text = file.read().decode("utf-8")
                resume_emb = model.encode(resume_text)
                score = cosine_similarity([job_emb], [resume_emb])[0][0]
                resume_scores.append((file.name, score))

            # Sort by score
            ranked = sorted(resume_scores, key=lambda x: x[1], reverse=True)

            # Toggle to show results
            with st.expander("Show Resume Ranking Results"):
                for name, score in ranked:
                    st.write(f"{name} — {round(score * 100, 2)}% match")