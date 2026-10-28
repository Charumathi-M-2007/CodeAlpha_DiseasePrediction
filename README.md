# Disease Prediction Using Machine Learning

## Overview

This project demonstrates a machine learning approach to classifying disease categories based on symptom-related input features. It uses a Random Forest Classifier to analyze symptom patterns and predict one of three predefined categories: Flu-like Illness, Common Cold, or Allergy.

The project covers the complete machine learning workflow, including dataset generation, data preprocessing, model training, evaluation, prediction, and visualization.

## Objectives

- Develop a machine learning model for symptom-based classification.
- Generate and process a structured dataset for training and testing.
- Train a Random Forest classification model.
- Evaluate model performance using standard classification metrics.
- Visualize classification results using a confusion matrix.
- Save the dataset and trained model for future experimentation.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Model Serialization:** Joblib

## Algorithm

### Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to perform classification. It is used in this project to learn patterns from the generated symptom data and predict predefined disease categories.

## Dataset

This project uses a synthetically generated educational dataset containing six binary symptom features:

- Fever
- Cough
- Headache
- Fatigue
- Sore throat
- Body pain

Each symptom is represented as `1` (present) or `0` (absent). The target variable contains one of three predefined categories.

**Important:** The dataset is artificially generated for demonstration purposes. It is not a verified medical dataset and does not represent real patient records.

## Project Structure

```text
DiseasePrediction/
├── disease_prediction.py
├── requirements.txt
├── README.md
├── disease_dataset.csv
├── disease_prediction_model.pkl
└── confusion_matrix.png
```

The CSV dataset, trained model, and confusion matrix are generated when the Python script runs successfully.

## Installation

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Execution

Run the following command:

```bash
python disease_prediction.py
```

The program generates the dataset, trains the classifier, evaluates performance, saves the model, and displays the confusion matrix.

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified test examples.
- **Precision:** Measures the correctness of predicted categories.
- **Recall:** Measures how many actual examples of each category are identified.
- **F1-Score:** Combines precision and recall into a single metric.
- **Confusion Matrix:** Displays correct and incorrect classifications for each category.

Actual performance metrics are displayed when the program is executed. Results should be interpreted only in the context of the synthetic dataset.

## Output

The project generates:

- Dataset preview and dataset dimensions.
- Model accuracy and classification report.
- Example symptom-based category prediction.
- `disease_dataset.csv` — generated dataset.
- `disease_prediction_model.pkl` — saved trained model.
- `confusion_matrix.png` — model evaluation visualization.

## Limitations

- The model is trained on synthetic data rather than verified clinical records.
- The predefined symptom patterns are artificial and may not reflect real medical relationships.
- The predicted category is not a medical diagnosis.
- The reported accuracy does not demonstrate real-world clinical performance.

## Future Enhancements

- Evaluate the model using reliable, appropriately licensed medical datasets.
- Compare multiple classification algorithms.
- Improve data validation and model evaluation.
- Develop an interactive interface for educational demonstrations.
- Add clear uncertainty communication and appropriate medical safety guidance.

## Learning Outcomes

This project provides practical experience in synthetic dataset generation, data handling, supervised machine learning, classification metrics, model serialization, and visualization using Python.

## Disclaimer

This project is intended solely for educational and demonstration purposes. It must not be used to diagnose diseases, guide treatment, or replace advice from a qualified healthcare professional.