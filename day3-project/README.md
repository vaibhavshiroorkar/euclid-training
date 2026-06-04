# Gym Member Segmentation using Unsupervised Machine Learning

## Project Overview

This project applies multiple unsupervised machine learning techniques to analyze and segment gym members based on their behavior and activity patterns. The goal is to identify natural groups of members, visualize these groups, and detect unusual behavior without using predefined labels.

The project utilizes clustering algorithms, dimensionality reduction, and anomaly detection techniques to gain insights into gym member characteristics.

---

## Dataset

**Dataset:** Gym Membership Dataset

The dataset contains information about gym members, including:

* Age
* Visits per Week
* Average Time in Gym
* Group Lesson Attendance
* Drink Subscription Status
* Personal Training Usage
* Sauna Usage

Dataset Size:

* 1000 Records
* 17 Features

---

## Objectives

* Preprocess and prepare gym member data
* Standardize features using StandardScaler
* Determine the optimal number of clusters using the Elbow Method
* Apply K-Means Clustering
* Evaluate clustering quality using:

  * Silhouette Score
  * Davies-Bouldin Index
* Perform Hierarchical Clustering
* Visualize hierarchical relationships using a Dendrogram
* Apply DBSCAN clustering
* Reduce dimensionality using PCA for visualization
* Detect anomalies using Isolation Forest
* Compare clustering results and interpret findings

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy
* Google Colab

---

## Machine Learning Techniques

### 1. K-Means Clustering

Partitions gym members into clusters based on similarity in behavior and gym usage patterns.

### 2. Hierarchical Clustering

Builds a hierarchy of clusters and visualizes member relationships through a dendrogram.

### 3. DBSCAN

Density-based clustering algorithm capable of identifying noise points and irregular member groups.

### 4. Principal Component Analysis (PCA)

Reduces dimensionality to two components for cluster visualization.

### 5. Isolation Forest

Detects anomalous gym members whose behavior significantly differs from the majority.

---

## Evaluation Metrics

### Silhouette Score

Measures how well data points fit within their assigned clusters.

* Range: -1 to 1
* Higher values indicate better clustering

### Davies-Bouldin Index

Measures cluster compactness and separation.

* Range: 0 to infinity
* Lower values indicate better clustering

---

## Workflow

1. Load Dataset
2. Data Preparation
3. Feature Selection
4. Standardization
5. Elbow Method
6. K-Means Clustering
7. Cluster Evaluation
8. Hierarchical Clustering
9. Dendrogram Visualization
10. DBSCAN Clustering
11. PCA Visualization
12. Isolation Forest Anomaly Detection
13. Result Comparison and Interpretation

---

## Results

The project successfully:

* Identified meaningful gym member segments
* Visualized clusters using PCA
* Compared multiple clustering algorithms
* Evaluated cluster quality using standard metrics
* Detected anomalous gym members using Isolation Forest

---

## Project Structure

```text
Gym-Member-Segmentation/
│
├── GymMemberSegmentation.ipynb
├── README.md
├── gym_membership.csv
└── images/
```

---

## Environment Setup for Contributors

Use a virtual environment before installing packages or running the project. This keeps dependencies isolated and makes the setup repeatable for future contributors.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install Project Dependencies

After activating the environment, install the libraries used by this project:

```bash
pip install -r requirements.txt
```

If you need to install manually, use the same package list from `requirements.txt`.

### Verify the Environment

```bash
python --version
pip list
```

When you are done working, deactivate the environment with:

```bash
deactivate
```

---

## Future Improvements

* Explore additional clustering algorithms
* Perform feature engineering
* Develop an interactive dashboard
* Apply advanced anomaly detection techniques
* Build member recommendation systems

---

## Author

Neil 
Vaibhav

Machine Learning Project – Unsupervised Learning using Clustering, PCA, and Anomaly Detection.