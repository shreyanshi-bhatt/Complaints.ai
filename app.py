import streamlit as st
from groq import Groq
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

# Initialize Groq API client
def load_groq_client():
    return Groq(api_key='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')  # Replace with your actual API key

# Create a prompt for classification
def create_prompt(text):
    # categories = ', '.join(complaint_categories)
    prompt = f"You are an expert in classifying complaints. Classify the following complaint into one of these categories: Product Quality Issues,Shipping and Delivery Problems,Customer Service Complaints,Payment and Billing Issues,Technical Issues,Product Availability,Marketing and Promotions,Privacy and Security Concerns,Return and Refund Process,Subscription Services .\n\nComplaint Text: {text}\n\nPlease provide only the category name as your response:"
    return prompt

# Classify the complaint using the Groq API
def classify_complaint(text, client):
    prompt = create_prompt(text)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192"  # Use the appropriate model name
    )
    result = chat_completion.choices[0].message.content
    return result.strip()

# Streamlit app
st.set_page_config(page_title="Complaints.ai")
st.title("Welcome to Complaints.ai")
st.write("In today's fast-paced world, sifting through countless customer complaints to identify key issues can be overwhelming. Complaints.ai tackles this challenge head-on with a cutting-edge LLM model that effortlessly processes and analyzes all your customer feedback. Our solution categorizes complaints into the 10 most common categories, including Product Quality Issues, Shipping and Delivery Problems, Customer Service Complaints, Payment and Billing Issues, Technical Issues, Product Availability, Marketing and Promotions, Privacy and Security Concerns, Return and Refund Process, and Subscription Services. To make the data even more actionable, Complaints.ai automatically generates a clear and concise bar graph, providing you with a powerful visual tool for better decision-making.")

# Load the Groq client once and store it in session state
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = load_groq_client()

groq_client = st.session_state.groq_client
st.subheader("File Upload")

uploaded_file = st.file_uploader("Upload a txt file containing a list of your customer complaints", type="txt")

if st.button("Analyze"):
    if uploaded_file is not None:
        complaints = uploaded_file.read().decode("utf-8").splitlines()

        # Initialize category counts
        category_counts = defaultdict(int)

        for complaint in complaints:
            category = classify_complaint(complaint, groq_client)
            if category in complaint_categories:
                category_counts[category] += 1

        # Display the results
        st.subheader("Complaint Counts per Category:")
        for category in complaint_categories:
            st.write(f"{category}: {category_counts.get(category, 0)}")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])

        # Display the bar chart
        st.subheader("Complaints Category Bar Chart")
        st.bar_chart(df.set_index('Category'))
