# Stroke Prediction using Weka
This project explores the use of machine learning algorithms for predicting stroke risk. We utilize the Weka libraries in Java to implement three classification models: Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN).

## Table of Contents
* [Project Goals](#project-goals)
* [Data](#data)
* [Algorithms](#algorithms)
* [Evaluation](#table-of-contents)
* [Instruction](#instruction)
* [Further Development](#further-development)



## Project Goal:

This project aims to develop models that can analyze patient data and predict the likelihood of a stroke occurrence.

## Data:

The project requires a dataset containing relevant patient attributes and a binary target variable indicating stroke presence/absence. The data should be formatted in the Attribute-Relation File Format (ARFF) compatible with Weka.

## Algorithms:

- Naive Bayes: This probabilistic classifier predicts the class (stroke or no stroke) based on the probability of each attribute value given the class.
- Decision Tree: This tree-like structure classifies data by following a series of decision rules based on attribute values.
- Random Forest
- Apriori

## Evaluation:

The performance of each model will be evaluated using metrics like accuracy, precision, recall, and F1-score. These metrics will help assess the model's effectiveness in predicting strokes.

## Instructions:

**Prerequisites:**
- Ensure you have Java installed.
- Download and install the Weka libraries from https://waikato.github.io/weka-wiki/.


**Data Preparation:**
- Obtain a stroke prediction dataset in ARFF format.
- Ensure the data contains relevant attributes and a binary target variable for stroke presence/absence.


## Further Development:

1. Explore additional machine learning algorithms like Support Vector Machines (SVM) or Random Forest.
2. erform feature selection techniques to identify the most impactful attributes for stroke prediction.
3. Implement techniques like cross-validation for robust model evaluation.
4. Visualize decision trees or other model structures for interpretability.

## *Disclaimer:*
This project is for educational purposes only and should not be used for real-world medical diagnosis. Always consult a qualified healthcare professional for medical advice.