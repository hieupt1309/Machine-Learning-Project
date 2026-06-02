# Machine-Learning-Project
# Comparative Analysis of Traditional and Transformer-based Text Representation Methods for Scientific Article Topic Classification

## 1. Project Overview

### Project Title

Comparative Analysis of Traditional and Transformer-based Text Representation Methods for Scientific Article Topic Classification

### Objective

This project aims to investigate and compare traditional machine learning methods and modern Transformer-based approaches for scientific article topic classification using article abstracts.

The project focuses on answering the following research questions:

* Which text representation method performs best?
* Which classification model achieves the highest accuracy?
* Does BERT significantly outperform traditional machine learning approaches?
* What is the trade-off between classification performance and computational cost?

---

# 2. Problem Definition

Given an abstract of a scientific paper, predict its primary research category.

Input:

* Scientific paper abstract

Output:

* Topic label

Examples:

* Computer Science (cs)
* Mathematics (math)
* Physics (physics)
* Statistics (stat)
* Quantitative Biology (q-bio)
* etc.

This is a multiclass text classification problem.

---

# 3. Dataset

Dataset Source:

UniverseTBD/arxiv-abstracts-large

Dataset contains scientific paper abstracts and their corresponding categories.

### Dataset Strategy

The original dataset contains approximately 38 categories.


### Select Top 15 Most Frequent Categories

Reason:

* More realistic than 5-class classification
* Less computationally expensive than 38-class classification
* Better balance between difficulty and training cost
* Allows clearer comparison between classical ML and Transformer models

### Sampling Strategy

For each selected category:4000 samples

Expected dataset size:60,000 abstracts
---

# 4. Project Pipeline

Raw Text
↓
Text Cleaning
↓
Train/Test Split
↓
Text Vectorization
↓
Model Training
↓
Hyperparameter Tuning
↓
Evaluation
↓
Comparison & Analysis

---

# 5. Text Preprocessing

The following preprocessing steps will be applied:

### 1. Lowercase

Example:

AI Is AMAZING

↓

ai is amazing

### 2. Remove Punctuation

Example:

hello!!!

↓

hello

### 3. Remove Numbers

Example:

AI in 2025

↓

ai in

### 4. Remove Stopwords

Example:

the cat is on the table

↓

cat table

### 5. Lemmatization

Examples:

running → run

studies → study

better → good

### Optional Improvements

* Rare word removal
* Phrase detection
* Context-aware stopword filtering

---

# 6. Text Representation Methods

The project compares three vectorization approaches.

## 6.1 TF-IDF

Advantages:

* Fast
* Lightweight
* Strong baseline

Disadvantages:

* Sparse representation
* Limited semantic understanding

---

## 6.2 Embedding-based Representation

Potential models:

* BAAI/bge-base-en-v1.5
* intfloat/e5-base-v2

Output dimension:

768

Advantages:

* Dense vectors
* Better semantic representation

---

## 6.3 Sentence-BERT (SBERT)

Model:

all-MiniLM-L6-v2

Alternative:

all-mpnet-base-v2

Advantages:

* Optimized for sentence embeddings
* Strong semantic understanding
* Expected to outperform TF-IDF

---

# 7. Models

## 7.1 Unsupervised Learning

### K-Means

Purpose:

* Clustering baseline

Evaluation:

* Accuracy (cluster-label mapping)
* Silhouette Score
* ARI

---

## 7.2 Classical Machine Learning Models

### Naive Bayes

Advantages:

* Very fast
* Strong baseline for text classification

---

### K-Nearest Neighbors (KNN)

Hyperparameters:

* k = 3
* k = 5
* k = 7

Expected performance:

* Works well with embeddings

---

### Logistic Regression

Advantages:

* Strong baseline
* Common benchmark for NLP tasks

---

### Support Vector Machine (SVM)

Advantages:

* One of the strongest classical models for text classification
* Often competitive with deep learning on medium-sized datasets

---

### Random Forest

Advantages:

* Ensemble of Decision Trees
* Better generalization than a single tree

---

### XGBoost

Advantages:

* Powerful boosting algorithm
* Frequently achieves state-of-the-art performance among classical ML methods

Expected performance:

* Best classical model

---

## 7.3 Deep Learning

### BERT Fine-Tuning

Model:

bert-base-uncased

Training configuration:

* Epochs: 3–5
* Batch Size: 16
* Max Length: 256

Expected performance:

* Highest overall accuracy

---

# 8. Research Hypotheses

## H1

SBERT > Embedding > TF-IDF

in terms of classification performance.

---

## H2

SVM and XGBoost will outperform other classical machine learning models.

---

## H3

BERT Fine-Tuning will achieve the highest overall performance.

---

## H4

TF-IDF + XGBoost may achieve competitive performance while requiring significantly less training time than BERT.

---

# 9. Evaluation Metrics

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score (Macro)
* F1-score (Weighted)

---

## Clustering Metrics

* Accuracy
* Silhouette Score
* Adjusted Rand Index (ARI)

---

## Computational Metrics

* Training Time
* Inference Time
* Memory Usage

---

## Visualization

* Confusion Matrix
* Class Distribution
* Accuracy Comparison Charts
* Training Time Comparison Charts

---

# 10. Experiments

The following combinations will be evaluated:

| Vectorization | Model               |
| ------------- | ------------------- |
| TF-IDF        | NB                  |
| TF-IDF        | KNN                 |
| TF-IDF        | Logistic Regression |
| TF-IDF        | SVM                 |
| TF-IDF        | Random Forest       |
| TF-IDF        | XGBoost             |
| TF-IDF        | KMeans              |
| Embedding     | NB                  |
| Embedding     | KNN                 |
| Embedding     | Logistic Regression |
| Embedding     | SVM                 |
| Embedding     | Random Forest       |
| Embedding     | XGBoost             |
| Embedding     | KMeans              |
| SBERT         | NB                  |
| SBERT         | KNN                 |
| SBERT         | Logistic Regression |
| SBERT         | SVM                 |
| SBERT         | Random Forest       |
| SBERT         | XGBoost             |
| SBERT         | KMeans              |
| Raw Text      | BERT Fine-Tuning    |

---

# 11. Expected Outcomes

Expected ranking:

BERT

>

XGBoost
≈
SVM

>

Random Forest

>

KNN

>

Naive Bayes

>

KMeans

Expected vectorization ranking:

SBERT

>

Embedding

>

TF-IDF

---

# 12. Project Structure

project/

├── data/

│ ├── raw/

│ ├── processed/

│

├── notebooks/

│

├── src/

│ ├── preprocessing/

│ ├── vectorization/

│ ├── models/

│ ├── evaluation/

│ └── utils/

│

├── results/

│ ├── figures/

│ ├── reports/

│

├── README.md

├── requirements.txt

└── main.py

---

# 13. Team Responsibilities 

## Member 1 — TF-IDF Representation

### Responsibilities

* Implement TF-IDF vectorization.
* Generate TF-IDF features for the dataset.
* Train and evaluate all machine learning models using TF-IDF representations.

### Models

* Naive Bayes
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* K-Means Clustering

### Deliverables

* Trained models
* Performance metrics
* Training and inference time
* Confusion matrices
* Experimental analysis for TF-IDF

---

## Member 2 — Dense Embedding Representation

### Responsibilities

* Generate dense embeddings using modern embedding models.

### Candidate Models

* BAAI/bge-base-en-v1.5
* intfloat/e5-base-v2

### Models

* Naive Bayes
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* K-Means Clustering

### Deliverables

* Embedding generation pipeline
* Trained models
* Evaluation results
* Experimental analysis for dense embeddings

---

## Member 3 — Sentence-BERT Representation

### Responsibilities

* Implement Sentence-BERT feature extraction.

### Candidate Models

* all-MiniLM-L6-v2
* all-mpnet-base-v2 (optional comparison)

### Models

* Naive Bayes
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* K-Means Clustering

### Deliverables

* SBERT embeddings
* Trained models
* Evaluation results
* Experimental analysis for SBERT

---

## Member 4 — BERT Fine-Tuning

### Responsibilities

* Fine-tune a Transformer-based classifier.

### Model

* bert-base-uncased

### Tasks

* Data preparation for BERT
* Tokenization
* Fine-tuning
* Hyperparameter tuning
* Model evaluation
* GPU training optimization

### Deliverables

* Fine-tuned BERT model
* Training logs
* Evaluation metrics
* Training and inference analysis

---

## Member 5 — Evaluation, Visualization & Presentation Preparation(Slide)

### Responsibilities

- Collect and organize experimental results from all team members.
- Build unified evaluation tables for model comparison.
- Create visualizations to analyze and present results.
- Compare the effectiveness of different text representation methods and machine learning models.
- Support report writing and prepare presentation materials.

### Tasks

- Result aggregation
- Evaluation metric analysis
- Comparative analysis
- Data visualization
- Report support
- Slide preparation

### Evaluation Metrics

#### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score (Macro)
- F1-Score (Weighted)

#### Clustering Metrics

- Accuracy
- Silhouette Score
- Adjusted Rand Index (ARI)

#### Computational Metrics

- Training Time
- Inference Time
- Memory Usage

### Visualizations

- Class Distribution
- Confusion Matrices
- Accuracy Comparison Charts
- Precision Comparison Charts
- Recall Comparison Charts
- F1-Score Comparison Charts
- Training Time Comparison
- Inference Time Comparison
- Memory Usage Comparison

### Deliverables

- Consolidated experimental results
- Evaluation tables
- Visualization figures
- Comparative analysis report
- Presentation slides
- Final project summary
# 14. Final Goal

Deliver a comprehensive empirical comparison between:

* Traditional feature engineering approaches
* Modern embedding-based approaches
* Transformer-based deep learning models

for scientific article topic classification.
