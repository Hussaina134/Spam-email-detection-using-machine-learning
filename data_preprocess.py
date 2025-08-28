import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("processed_data.csv")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split()) 
    return text

# Apply cleaning
df['clean_text'] = df['message'].apply(clean_text)

# Features and labels
X = df['clean_text']
y = df['label'].astype(int)

# Convert text → numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# --------------------------
# Define Models
# --------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    # "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results = {}


if __name__ == "__main__":
    results = {}
# --------------------------
# Train & Evaluate
# --------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# --------------------------
# Pick Best Model
# --------------------------
best_model_name = max(results, key=lambda x: results[x]['F1'])
best_model = models[best_model_name]

print("\nBest Model:", best_model_name, results[best_model_name])

# Save best model and vectorizer
pickle.dump(best_model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
# Save model info
model_info = {
    "best_model_name": best_model_name,
    "dataset_shape": df.shape,
    "evaluation": results[best_model_name]  # metrics for best model
}

pickle.dump(model_info, open("model_info.pkl", "wb"))

