# 📧 Email Spam Detection using Machine Learning

This project focuses on detecting **spam emails** using supervised machine learning techniques.  
We built and trained multiple ML models on the **TREC07p dataset** and deployed an interactive web app using **Streamlit** for real-time email classification.

---

## 🔹 Overview
Email spam is one of the most common cyber threats, leading to phishing, fraud, and malware attacks.  
This project applies machine learning algorithms to classify emails as **Spam** or **Ham (Not Spam)**.

---

## 📂 Dataset
- **Source:** [TREC 2007 Spam Corpus](http://plg.uwaterloo.ca/~gvcormac/trec07p/)  
- **Size:** ~75,000 emails  
- **Preprocessing applied:**
  - Tokenization
  - Stopword removal
  - TF-IDF Vectorization
  - Train-Test split (80:20)

---

## ⚙️ Methodology
1. Data Preprocessing (cleaning + vectorization)  
2. Model Training with multiple ML algorithms:
   - Logistic Regression  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - Naïve Bayes  
3. Model Evaluation using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
4. Best-performing model deployed in a **Streamlit app**.

---

## 📊 Results
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 97.5%    | 97%       | 96%    | 96.5%    |
| Random Forest        | 96.2%    | 95%       | 94%    | 94.5%    |
| SVM                  | 97.0%    | 96%       | 95%    | 95.5%    |
| Naïve Bayes          | 94.8%    | 93%       | 92%    | 92.5%    |

✅ **Logistic Regression** gave the best results.

---

## 🚀 Streamlit Web App
I created an interactive UI where users can input an email and instantly check whether it is **Spam or Not Spam**.

### Run Locally
```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install -r requirements.txt
streamlit run app.py
