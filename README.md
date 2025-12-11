# Breast Cancer Diagnosis Prediction

A machine learning project that classifies breast tumors as **Malignant** or **Benign** using the Wisconsin Breast Cancer Dataset.

## Approach

### 1. Data Preprocessing
- Loaded the dataset (569 samples, 30 features)
- Encoded diagnosis labels: Malignant (M) → 1, Benign (B) → 0
- Removed non-predictive columns (`id`, `Unnamed: 32`)
- Applied **Min-Max Normalization** to scale all features to [0, 1]
- Split data into 80% training / 20% testing (stratified)

### 2. Unsupervised Learning: K-Means Clustering
- Applied K-Means with k=2 clusters to discover natural groupings
- Mapped clusters to majority class labels
- Used to validate that the data has separable structure

### 3. Supervised Learning: K-Nearest Neighbors (KNN)
- Trained KNN classifier with k=5 neighbors
- Used Euclidean distance on normalized features
- Evaluated on held-out test set

## Results

| Metric | Value |
|--------|-------|
| **KNN Accuracy** | 96.49% |
| **Precision** | 100% |
| **Recall** | 90.48% |
| **F1-Score** | 95.00% |
| **K-Means Clustering Accuracy** | 92.79% |

### Confusion Matrix
|  | Predicted Benign | Predicted Malignant |
|--|------------------|---------------------|
| **Actual Benign** | 72 | 0 |
| **Actual Malignant** | 4 | 38 |

## Key Findings
- KNN achieved high precision (no false positives for malignant)
- 4 malignant cases were misclassified as benign (false negatives)
- K-Means clustering alone achieved ~93% accuracy, showing clear data separability

## Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## How to Run
1. Clone this repository
2. Ensure `Dataset/Dataset.csv` is present
3. Open and run `Notebook/notebook.ipynb`

## Project Structure
```
├── Dataset/
│   └── Dataset.csv
├── Notebook/
│   ├── notebook.ipynb
│   └── breast_cancer_assignment_output/
└── README.md
```
