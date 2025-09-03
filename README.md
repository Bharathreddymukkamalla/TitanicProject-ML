
# Titanic Survival Prediction 

This project predicts passenger survival on the **Titanic dataset** using **Logistic Regression**.  
It is implemented in **Python** with **Google Colab**, using data preprocessing, visualization, and model evaluation.

---

## Dataset
The dataset used is the **Kaggle Titanic: Machine Learning from Disaster** dataset:  
- Train.csv contains details like Passenger Class, Sex, Age, Fare, Siblings/Spouses, Parents/Children, Embarked Port, etc.  
- Target column: **Survived** (`1 = survived`, `0 = did not survive`)

Shape of dataset: **891 rows × 12 columns**

---

## Dependencies
Install the required libraries before running the notebook:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
[train.csv](https://github.com/user-attachments/files/22119013/train.csv)
 Steps in the Project
1. Data Collection & Preprocessing
Loaded dataset (train.csv) into Pandas DataFrame.

Checked null values and handled missing data:

Dropped Cabin column.

Filled missing Age with mean.

Filled missing Embarked with mode (most frequent value).

Encoded categorical columns:

Sex → male = 0, female = 1

Embarked → S = 0, C = 1, Q = 2

2. Data Analysis & Visualization
Countplots of Survived vs Non-Survived.

Gender-based survival comparison.

Passenger class and Embarked port analysis with survival.

3. Feature Selection
Dropped unnecessary columns: PassengerId, Name, Ticket.

Features (X): Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Target (y): Survived

4. Model Training
Split data into training (80%) and testing (20%).

Applied Logistic Regression.

5. Model Evaluation
Training Accuracy: 80.75%

Test Accuracy: 78.21%

The model performs reasonably well on unseen data, showing generalization capability.

* Results
Logistic Regression achieves around 78% accuracy on test data.

Gender, passenger class, and age strongly affect survival chances.

* How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/titanic-survival-prediction.git
Open the notebook in Google Colab or Jupyter Notebook.

Upload the Titanic dataset (train.csv).

Run all cells to train and evaluate the model.

* Future Improvements
Try other ML models (Random Forest, SVM, XGBoost).

Feature engineering (family size, title extraction from names).

Hyperparameter tuning with GridSearchCV.

🏷 Author
Developed by Bharath reddy Mukkamalla
