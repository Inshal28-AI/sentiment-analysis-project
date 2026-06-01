import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# LOAD DATA
df = pd.read_csv("data/twitter_training.csv")

df = df.iloc[:, [2, 3]]
df.columns = ["label", "text"]

df["label"] = df["label"].str.lower()
df = df[df["label"].isin(["positive", "negative", "neutral"])]

df["label"] = df["label"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

model = LinearSVC()
model.fit(X_train_vec, y_train)

# SAVE MODEL
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_test_vec = vectorizer.transform(X_test)

y_pred = model.predict(X_test_vec)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()