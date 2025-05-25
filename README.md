# An AI-driven dual phase model for heart disease diagonisation
🫀 Heart Disease Detection Model
This project presents a Machine Learning-based Heart Disease Detection System that predicts the likelihood of a person having heart disease based on key clinical features. The model is trained and evaluated on a well-known heart disease dataset, and aims to support early diagnosis, improve decision-making, and ultimately contribute to better patient outcomes.

📌 Objective
The main objective of this project is to develop a predictive model that can accurately detect the presence of heart disease using machine learning algorithms. This tool can be useful for medical professionals and researchers to assist in preliminary diagnoses and risk assessment.

🧠 Machine Learning Models Used
Multiple classification models were explored and evaluated to determine the best-performing algorithm:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree

🌲 XGBoost Algorithm – Explained
XGBoost (Extreme Gradient Boosting) is a highly efficient and scalable implementation of gradient boosting, developed by Tianqi Chen. It's widely used in structured/tabular machine learning competitions and real-world applications due to its speed and performance.

📌 What Is Gradient Boosting?
Before diving into XGBoost, it's helpful to understand gradient boosting:

Boosting is an ensemble learning technique that builds models sequentially.

Each new model attempts to correct the errors made by the previous models.

Gradient Boosting builds trees one at a time, and each tree tries to minimize the residual (error) of the previous tree using gradient descent.

⚙️ How XGBoost Works
XGBoost improves on traditional gradient boosting with several innovations:

1. Objective Function
The objective function in XGBoost combines two parts:

ini
Copy
Edit
Obj = Loss function + Regularization term
Loss Function: Measures how well the model fits the data (e.g., logistic loss for classification, squared error for regression).

Regularization: Penalizes model complexity to prevent overfitting.

2. Additive Tree Building
Instead of training all trees at once, XGBoost builds them one at a time. For each tree:

It calculates the gradient (1st derivative) and hessian (2nd derivative) of the loss function.

These values guide how the tree should split and how much weight each leaf should carry.

3. Regularization (Shrinkage and Pruning)
Shrinkage (learning rate): Reduces the influence of each new tree by multiplying predictions by a learning rate (e.g., 0.1). This slows down the learning and helps generalize better.

Pruning: Unlike traditional decision trees that grow greedily, XGBoost grows the tree and then prunes back branches that do not improve the objective.

4. Column Subsampling
XGBoost can randomly sample features (columns) for each tree, like in Random Forests. This adds diversity and helps avoid overfitting.

5. Handling Missing Values
XGBoost can learn how to handle missing data by assigning default directions when a value is missing, based on training loss reduction.

6. Parallel and Distributed Computing
Unlike traditional boosting, which is sequential and slow, XGBoost implements:

Parallel computation of feature splits,

Distributed training for large datasets,

Cache optimization for better memory and CPU usage.

🔢 Mathematical Formulation
For a dataset with n examples and m features, and prediction at round t as:

Copy
Edit
ŷᵢ^(t) = ŷᵢ^(t-1) + fₜ(xᵢ)
Where:

fₜ(xᵢ) is the prediction from the t-th tree,

The objective becomes:

Copy
Edit
Obj^(t) = Σ₁ⁿ l(yᵢ, ŷᵢ^(t-1) + fₜ(xᵢ)) + Ω(fₜ)
Using Taylor approximation (2nd order) to simplify and optimize.

✅ Advantages of XGBoost
🚀 High speed and performance

🔍 Built-in regularization (L1 & L2)

🧠 Handles missing values automatically

⚖️ Supports both classification and regression

📊 Feature importance scoring

🌐 Scalable to large datasets

❌ Limitations
Can overfit if not tuned properly

Requires careful hyperparameter tuning

Less interpretable than simpler models (but tools like SHAP help)

🔧 Common Hyperparameters
Parameter	Description
n_estimators	Number of boosting rounds
learning_rate	Step size shrinkage
max_depth	Maximum depth of a tree
subsample	Fraction of samples to be used per tree
colsample_bytree	Fraction of features to be used per tree
gamma	Minimum loss reduction to make a split
lambda, alpha	L2 and L1 regularization
objective	Loss function (e.g., binary:logistic, reg:squarederror)

🧪 Use in Heart Disease Prediction
In your project, XGBoost can be used as a powerful classifier to predict whether a patient is at risk of heart disease, based on clinical features. It can achieve high accuracy and robustness, especially when trained with proper cross-validation and hyperparameter tuning.



After comparing performance metrics such as accuracy, precision, recall, and F1-score, the best model was selected for deployment.

📊 Dataset
The dataset used is the Cleveland Heart Disease dataset, which is publicly available via the UCI Machine Learning Repository. It includes 14 key medical attributes:

age: Age of the patient

sex: Gender (1 = male, 0 = female)

cp: Chest pain type (0 to 3)

trestbps: Resting blood pressure

chol: Serum cholesterol in mg/dl

fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

restecg: Resting electrocardiographic results (0 to 2)

thalach: Maximum heart rate achieved

exang: Exercise induced angina (1 = yes; 0 = no)

oldpeak: ST depression induced by exercise

slope: Slope of the peak exercise ST segment

ca: Number of major vessels (0–3) colored by fluoroscopy

thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)

target: Diagnosis of heart disease (1 = disease present; 0 = no disease)

⚙️ Workflow
Data Preprocessing

Handling missing values

Feature encoding and scaling

Correlation analysis and feature selection

Model Training

Data split into training and testing sets

Hyperparameter tuning using GridSearchCV 

Training multiple models and evaluating their performance

Model Evaluation

Accuracy, precision, recall, F1-score

Confusion matrix

ROC-AUC curve

A simple Streamlit or Flask-based web app is used for user interaction

🚀 Features
High accuracy in prediction with minimal input

User-friendly model ready for integration

Modular code for easy experimentation and extension



🛠️ Technologies Used
Python

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

Jupyter Notebook / Google Colab

Streamlit 

📈 Results
The final selected model achieved:

Accuracy: ~91%

Precision: ~91%

Recall: ~69%

F1-Score: ~69%

(📌 Replace 96.44% with your actual results)
