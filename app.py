
import streamlit as st
import PyPDF2
import pickle
import re
import string
import pandas as pd
import plotly.express as px

# --------------------------
# Load trained model, vectorizer, and model info
# --------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model_info = pickle.load(open("model_info.pkl", "rb"))

# --------------------------
# Text cleaning function
# --------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")
st.title("📧 Email Spam Detector")
st.write("Upload one or more PDF emails, and the model will classify them as SPAM or NOT SPAM.")

# --------------------------
# Show Model Summary
# --------------------------
st.subheader("Model Summary")
st.write(f"**Algorithm Used:** {model_info['best_model_name']}")
st.write("**Evaluation Metrics (Best Model):**")
st.write(f"- Accuracy: {model_info['evaluation']['Accuracy']:.4f}")
st.write(f"- Precision: {model_info['evaluation']['Precision']:.4f}")
st.write(f"- Recall: {model_info['evaluation']['Recall']:.4f}")
st.write(f"- F1 Score: {model_info['evaluation']['F1']:.4f}")

# --------------------------
# File Upload
# --------------------------
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        result = "SPAM 🚫" if prediction == 1 else "NOT SPAM ✅"
        
        results.append({
            "filename": uploaded_file.name,
            "result": result
        })

    # Display results table
    st.subheader("Classification Results")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

# --------------------------
# Horizontal bar chart with Plotly
# --------------------------
    st.subheader("Spam vs Not Spam Count (Interactive)")

    counts = df_results['result'].value_counts().reset_index()
    counts.columns = ['Result', 'Count']
    counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)

    fig = px.bar(
        counts,
        x='Count',
        y='Result',
        orientation='h',
        text='Count',
        color='Result',
        color_discrete_map={"SPAM 🚫": "red", "NOT SPAM ✅": "green"},
        labels={'Count':'Number of Emails', 'Result':'Classification'}
        )
    fig.update_traces(texttemplate='%{text} (%{customdata[0]}%)', textposition='outside', customdata=counts[['Percentage']])
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


