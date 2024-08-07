import streamlit as st
from langchain_community.llms import CTransformers
from collections import defaultdict
import pandas as pd

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


# Create a prompt for classification
def create_prompt(text):
    categories = ', '.join(complaint_categories)
    prompt = f"Classify the following complaint into one of these categories: {categories}\n\nComplaint Text: {text}\n\nCategory:"
    return prompt


# Classify the complaint using the model
def classify_complaint(text, model):
    prompt = create_prompt(text)
    result = model(prompt)
    return result.strip()


# Load the model once and store in session state
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )


# Streamlit app
st.title("Complaint Classification App")

# Load the model once and store it in session state
if 'model' not in st.session_state:
    st.session_state.model = load_llm()

model = st.session_state.model

uploaded_file = st.file_uploader("Upload a complaints.txt file", type="txt")

if st.button("Analyze"):
    if uploaded_file is not None:
        complaints = uploaded_file.read().decode("utf-8").splitlines()

        # Initialize category counts
        category_counts = defaultdict(int)

        for complaint in complaints:
            category = classify_complaint(complaint, model)
            if category in complaint_categories:
                category_counts[category] += 1

        # Display the results
        st.write("Complaint Counts per Category:")
        for category in complaint_categories:
            st.write(f"{category}: {category_counts.get(category, 0)}")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])

        # Create a Streamlit app
        st.write("Complaints Category Bar Chart")

        # Display the bar chart
        st.bar_chart(df.set_index('Category'))

