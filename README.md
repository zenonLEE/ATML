# 🧠 ATML Text Classifier

**"A Novel Framework for Adsorption Thermodynamics: Combining Standardized Methodology with Machine Learning-based Text Classification"**

This repository contains the implementation of the machine learning-based text classification component described in the above paper. The classifier is designed to assist in selecting the appropriate theoretical framework for adsorption thermodynamics based on textual input.

It distinguishes between the following three adsorption system assumptions:

- **Classical Closed System**
- **Polanyi Open System**
- **Modern Open System**

---

## 🚀 Features

- TF-IDF + Naive Bayes classification pipeline
- Paragraph-level data expansion for small sample learning
- Leave-One-Out Cross-Validation (LOOCV) for rigorous evaluation
- Optional independent test set evaluation
- Confidence scoring and detailed probability output
- Reusable classification function

---

## 🗂️ File Structure

atml_text_classifier.py # Main pipeline: training, evaluation, prediction s1.txt, s2.txt, s3.txt # Core training texts for each scenario test.txt # Optional test set for external evaluation README.md # Project documentation

yaml
Copy
Edit

---

## 🧪 Example: Classifying a Sample

```python
from atml_text_classifier import classify_text

text = "The adsorption equilibrium behavior suggests a partially open system ..."
label, confidence = classify_text(text, print_details=True)

print(f"Predicted Class: {label}")
print(f"Confidence: {confidence:.4f}")
```python

⚙️ Dependencies
Install required packages:

bash
Copy
Edit
pip install numpy scikit-learn
Tested with:

Python ≥ 3.8

scikit-learn ≥ 1.3

NumPy ≥ 1.22
