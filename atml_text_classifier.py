import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# 1. Load training texts
with open("s1.txt", "r", encoding="utf-8") as f:
    s1_text = f.read()
with open("s2.txt", "r", encoding="utf-8") as f:
    s2_text = f.read()
with open("s3.txt", "r", encoding="utf-8") as f:
    s3_text = f.read()

# 2. Build base training set
texts = [s1_text, s2_text, s3_text]
labels = ["Classical Closed System", "Polanyi Open System", "Modern Open System"]

# 3. Extract paragraphs from each document to expand training data
def extract_paragraphs(text, min_length=50, max_paragraphs=10):
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) >= min_length]
    return paragraphs[:max_paragraphs]

# Expand dataset
extended_texts = []
extended_labels = []

for i, text in enumerate(texts):
    extended_texts.append(text)
    extended_labels.append(labels[i])
    paragraphs = extract_paragraphs(text)
    for para in paragraphs:
        extended_texts.append(para)
        extended_labels.append(labels[i])

print(f"Expanded dataset size: {len(extended_texts)}")
for label in set(extended_labels):
    count = extended_labels.count(label)
    print(f"Class '{label}': {count} samples")

# 4. Train model on all data
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000)),
    ("clf", MultinomialNB(alpha=0.1))
])
model.fit(extended_texts, extended_labels)

# 5. Evaluation Method 1: Leave-One-Out Cross-Validation
loo = LeaveOneOut()
predictions = []
true_labels = []

for train_idx, test_idx in loo.split(extended_texts):
    X_train = [extended_texts[i] for i in train_idx]
    y_train = [extended_labels[i] for i in train_idx]
    X_test = [extended_texts[i] for i in test_idx]
    y_test = [extended_labels[i] for i in test_idx]

    temp_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("clf", MultinomialNB(alpha=0.1))
    ])
    temp_model.fit(X_train, y_train)
    pred = temp_model.predict(X_test)[0]

    predictions.append(pred)
    true_labels.append(y_test[0])

# Evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions,
    average='weighted',
    zero_division=1
)

print("\nLeave-One-Out Cross-Validation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(true_labels, predictions)
print(conf_matrix)

print("\nClassification Report:")
report = classification_report(true_labels, predictions, zero_division=1)
print(report)

print(f"\nPseudo R^2 (Accuracy): {accuracy:.4f}")

# 6. Evaluation Method 2: Independent Test Set
try:
    with open("test.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    test_samples = []
    test_labels = []
    lines = test_text.split("\n")
    has_labels = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            parts = line.split("|", 1)
            test_samples.append(parts[0].strip())
            test_labels.append(parts[1].strip())
            has_labels = True
        elif "\t" in line:
            parts = line.split("\t", 1)
            test_samples.append(parts[0].strip())
            test_labels.append(parts[1].strip())
            has_labels = True
        else:
            test_samples.append(line)

    if not has_labels and len(test_samples) == 0:
        test_samples = [test_text]

    test_predictions = model.predict(test_samples)
    test_probas = model.predict_proba(test_samples)

    print("\nIndependent Test Set Results:")

    if has_labels:
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions,
            average='weighted',
            zero_division=1
        )

        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {test_f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(test_labels, test_predictions, zero_division=1))
        print(f"\nPseudo R^2 (Accuracy): {test_accuracy:.4f}")

    print("\nPrediction Results:")
    for i, (sample, prediction) in enumerate(zip(test_samples, test_predictions)):
        confidence = max(test_probas[i])
        preview = sample[:100] + "..." if len(sample) > 100 else sample
        print(f"\nSample {i + 1}:")
        print(f"Preview: {preview}")
        print(f"Predicted Class: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print("Class Probabilities:")
        for j, cls in enumerate(model.classes_):
            print(f"  - {cls}: {test_probas[i][j]:.4f}")
        if has_labels:
            print(f"True Class: {test_labels[i]}")
            print(f"Correct: {'✓' if prediction == test_labels[i] else '✗'}")

except FileNotFoundError:
    print("\nWarning: 'test.txt' not found. Skipping test set evaluation.")
except Exception as e:
    print(f"\nError during test set evaluation: {str(e)}")


# 7. Prediction Function
def classify_text(text, print_details=False):
    predicted = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = max(proba)

    if print_details:
        classes = model.classes_
        for i, cls in enumerate(classes):
            print(f"  - {cls}: {proba[i]:.4f}")

    return predicted, confidence
