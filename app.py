import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the assessment data
df = pd.read_csv("shl_assessments.csv")

# Combine fields for embedding
df["text_for_embedding"] = df["Name"] + " " + df["Description"] + " " + df["Test Type"]

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Pre-compute catalog embeddings
catalog_embeddings = model.encode(df["text_for_embedding"].tolist(), convert_to_tensor=True)

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation Engine")
query = st.text_area("Paste a job description or type your hiring needs here:")

if st.button("Get Recommendations"):
    if query.strip():
        query_embedding = model.encode(query, convert_to_tensor=True)

        similarities = cosine_similarity(
            [query_embedding.cpu().numpy()], catalog_embeddings.cpu().numpy()
        )[0]

        top_indices = similarities.argsort()[-5:][::-1]

        results = df.iloc[top_indices].copy()
        results["Similarity Score"] = similarities[top_indices].round(2)

        # Make Assessment column clickable
        results["Assessment"] = results.apply(
            lambda x: f"[{x['Name']}]({x['URL']})", axis=1
        )

        # Rename columns
        results.rename(columns={
            "Remote": "Remote Support",
            "Adaptive": "Adaptive Support",
            "Duration": "Duration (min)",
            "Test Type": "Test Category"
        }, inplace=True)

        # Reorder columns
        results = results[[
            "Assessment",
            "Remote Support",
            "Adaptive Support",
            "Duration (min)",
            "Test Category",
            "Similarity Score"
        ]]

        st.markdown("### ✅ Top Recommendations:")
        st.markdown(results.to_markdown(index=False), unsafe_allow_html=True)

    else:
        st.warning("Please enter a query or job description first.")
