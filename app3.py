import streamlit as st
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import pandas as pd

# Load the model for embeddings
def load_embedding_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # A smaller, faster model

# Define complaint categories
complaint_categories = [
    'Product Quality Issues',
    'Shipping and Delivery Problems',
    'Customer Service Complaints',
    'Payment and Billing Issues',
    'Technical Issues',
    'Product Availability',
    'Marketing and Promotions',
    'Privacy and Security Concerns',
    'Return and Refund Process',
    'Subscription Services'
]

# Streamlit app
st.title("Complaint Classification App")

# Load the embedding model once and store it in session state
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = load_embedding_model()

embedding_model = st.session_state.embedding_model

# Precompute category embeddings
category_embeddings = embedding_model.encode(complaint_categories)

uploaded_file = st.file_uploader("Upload a complaints.txt file", type="txt")

if st.button("Analyze"):
    if uploaded_file is not None:
        complaints = uploaded_file.read().decode("utf-8").splitlines()

        # Initialize category counts
        category_counts = defaultdict(int)

        for complaint in complaints:
            complaint_embedding = embedding_model.encode(complaint)
            # Calculate cosine similarities
            similarities = util.cos_sim(complaint_embedding, category_embeddings)
            # Get the index of the highest similarity
            best_category_idx = similarities.argmax()
            best_category = complaint_categories[best_category_idx]
            category_counts[best_category] += 1

        # Display the results
        st.subheader("Complaint Counts per Category:")
        for category in complaint_categories:
            st.write(f"{category}: {category_counts.get(category, 0)}")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])

        # Display the bar chart
        st.subheader("Complaints Category Bar Chart")
        st.bar_chart(df.set_index('Category'))
