# Version 1

import streamlit as st
import gensim
from gensim import corpora, models
import re
import heapq

# Define keywords for different complaint categories
complaints = {
    'Product Quality Issues': [
        "defective", "damaged", "poor quality", "malfunction", "faulty", "substandard", "broken",
        "unreliable", "shoddy", "flawed", "incorrect", "inaccurate", "not as described", "low quality",
        "deteriorated", "uneven", "inferior", "unsatisfactory", "problematic", "nonfunctional",
        "inoperative", "mismatch", "unfit", "fail", "dud", "cracked", "incomplete", "scratched", "misaligned"
    ],
    'Shipping and Delivery Problems': [
        "late", "delayed", "missing", "wrong address", "lost", "damaged during shipping", "not delivered",
        "shipment error", "incorrect package", "delivered to wrong address", "lost in transit",
        "late delivery", "failed delivery", "poor handling", "shipping error", "delivered damaged",
        "undelivered", "return issue", "shipment discrepancy", "lost package", "incorrect delivery",
        "shipping delay", "delivery problem", "late arrival", "dispatched incorrectly", "misdelivered",
        "unreliable", "incomplete order", "package lost"
    ],
    'Customer Service Complaints': [
        "unresponsive", "rude", "unhelpful", "slow response", "poor service", "inattentive", "unprofessional",
        "inefficient", "disrespectful", "unfriendly", "unavailable", "untrained", "misleading",
        "uncooperative", "incorrect information", "poor communication", "arrogant", "failure to assist",
        "bad attitude", "dismissive", "incompetent", "ignoring", "unresolved issue", "poor support",
        "lack of empathy", "unaccommodating", "delayed response", "mismanagement", "unresponsive staff"
    ],
    'Payment and Billing Issues': [
        "overcharged", "undercharged", "billing error", "incorrect charge", "fraudulent charge", "refund issue",
        "payment processing error", "charge dispute", "unauthorized charge", "duplicate charge",
        "incorrect invoice", "billing discrepancy", "payment failed", "missing payment", "credit issue",
        "payment problem", "unpaid balance", "discrepancy", "account charge", "error in billing",
        "invoice error", "refund delay", "payment issue", "chargeback", "incorrect payment", "billing mistake",
        "payment error", "late charge", "refund request", "credit discrepancy", "disputed charge"
    ],
    'Technical Issues': [
        "software bug", "crash", "error message", "malfunction", "glitch", "buggy", "slow performance",
        "incompatibility", "network issue", "connection problem", "system failure", "downtime", "faulty update",
        "data loss", "technical error", "software failure", "hardware issue", "unresponsive", "freezing",
        "corrupted data", "install issue", "configuration problem", "reboot required", "system error",
        "update issue", "technical glitch", "crashed", "uninstall issue", "device issue", "sync problem",
        "failure to load"
    ],
    'Product Availability': [
        "out of stock", "unavailable", "discontinued", "not in stock", "backordered", "inventory issue",
        "limited stock", "supply issue", "stock shortage", "no longer available", "not restocked", "stock-out",
        "product shortage", "inventory shortage", "not in inventory", "sold out", "product discontinuation",
        "availability issue", "short supply", "limited availability", "not on shelf", "restocking delay",
        "unreachable", "out of inventory", "stock unavailability", "discontinued item", "missing from store",
        "low inventory", "not available", "supply chain issue"
    ],
    'Marketing and Promotions': [
        "misleading ad", "false promotion", "unfulfilled offer", "ad discrepancy", "expired coupon",
        "promotion not honored", "incorrect discount", "deceptive marketing", "invalid offer",
        "error in promotion", "misrepresentation", "ad error", "unclaimed promotion", "discount issue",
        "incorrect advertisement", "promotion problem", "offer not applied", "promotional error",
        "inaccurate ad", "not as advertised", "unavailable deal", "marketing mistake", "false advertising",
        "promotional miscommunication", "coupon issue", "inconsistent offer", "ad problem", "wrong promotion",
        "misrepresented offer", "discount discrepancy"
    ],
    'Privacy and Security Concerns': [
        "data breach", "privacy violation", "unauthorized access", "security lapse", "data leak",
        "confidentiality issue", "personal information exposed", "security breach", "privacy issue",
        "data protection failure", "unauthorized use", "information theft", "security concern", "privacy breach",
        "data misuse", "compromised data", "privacy violation", "hacked account", "data theft",
        "unauthorized disclosure", "security failure", "personal data breach", "insecure system",
        "privacy infringement", "data loss", "sensitive data exposure", "identity theft",
        "data security issue", "confidential information leaked", "data compromise", "security flaw"
    ],
    'Return and Refund Process': [
        "difficult return", "slow refund", "denied return", "return policy issue", "refund delay",
        "return problem", "refund not processed", "policy confusion", "return request issue",
        "refund discrepancy", "complicated return", "incomplete refund", "return error",
        "refund process issue", "policy enforcement", "return shipping issue", "refund not received",
        "return authorization", "return handling", "refund request delay", "return label issue",
        "complicated process", "refund problem", "return dispute", "refund error", "return shipping delay",
        "issue with return", "return policy confusion", "ineligible for refund", "return not accepted",
        "refund claim issue"
    ],
    'Subscription Services': [
        "subscription cancellation", "billing error", "auto-renewal issue", "service interruption",
        "subscription problem", "account access issue", "subscription charge", "unwanted renewal",
        "plan change issue", "subscription error", "cancellation problem", "refund issue",
        "account billing", "subscription discrepancy", "service not as described", "billing discrepancy",
        "subscription fee", "plan not available", "account issue", "auto-renewal problem", "service charge",
        "subscription confusion", "cancellation delay", "billing mistake", "subscription not honored",
        "plan change problem", "unwanted charge", "service disruption", "subscription not updated",
        "access issue"
    ]
}


def label_topic(text):
    """
    Given a piece of text, this function returns the top two complaint labels that best match the topics discussed in the text.
    """
    # Initialize counts
    counts = {category: 0 for category in complaints.keys()}

    # Count the number of occurrences of each keyword in the text for each complaint category
    for category, keywords in complaints.items():
        count = sum(
            [len(re.findall(r"\b{}\b".format(re.escape(keyword)), text, re.IGNORECASE)) for keyword in keywords])
        counts[category] = count

    # Get the top two complaints based on their counts
    top_complaints = heapq.nlargest(2, counts, key=counts.get)

    return top_complaints if counts[top_complaints[0]] > 0 else []


def preprocess_text(text):
    """
    Preprocess the text by tokenizing and removing stop words.
    """
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    return [token for token in tokens if token not in stop_words]


def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    """
    Perform topic modeling using LDA.
    """
    preprocessed_text = [preprocess_text(transcript_text)]
    dictionary = corpora.Dictionary(preprocessed_text)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics


# Streamlit app
st.title("AI-Powered Complaints Analyzer")

# Upload transcript file
uploaded_file = st.file_uploader("Upload Transcript File", type=["txt"])

# Initialize count list
count_list = {category: 0 for category in complaints.keys()}

if st.button("Analyze"):
    if uploaded_file is not None:
        # Read the transcript text
        transcript_text = uploaded_file.read().decode("utf-8")

        # Analyze and categorize the transcript
        topic_labels = label_topic(transcript_text)
        for label in topic_labels:
            count_list[label] += 1

        # Display topic analysis
        st.write("Categorized Complaints:")
        st.write(count_list)

