# Interview Acceptance Prediction Project

The rising competitiveness in professional hiring processes highlights the importance of early prediction and candidate assessment. This project implements and compares three machine learning classification algorithms—Logistic Regression, Naive Bayes, and K-Nearest Neighbors (KNN)—to predict whether an applicant will be accepted for an interview based on professional and personal background features, helping streamline recruitment pipelines and demonstrating the practical application of machine learning in HR analytics.

1. **Dataset Description & Preprocessing**
   - **File:** `logatta.csv` (placed in the same local directory).
   - **Features Used:** Age, BusinessTravel, Education, MaritalStatus, OverTime, EmployeeNumber, DailyRate, and Gender.
   - **Target Variable:** `accepted for the interview` (Binary: 0 for Rejected, 1 for Accepted).
   - **Preprocessing Steps:** 
     - Categorical text features were converted into numerical format using One-Hot Encoding via `pd.get_dummies`.
     - Target boolean values were mapped to binary integers (`0` and `1`).
     - Features were normalized using `StandardScaler` to ensure optimal performance for distance-based and linear models.

2. **How to Get Started**
   - Clone the repository and switch to your feature branch:
     ```bash
     git checkout -b feature/interview-prediction
     ```
   - Ensure you have Python and Jupyter Notebook installed along with required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
   - Place the `logatta.csv` dataset file in the exact same directory as your Jupyter Notebook.

3. **How to Run and Test the Project**
   - Open your Jupyter Notebook environment.
   - Ensure `logatta.csv` is located in the working directory.
   - Run all code cells sequentially from top to bottom.
   - The script will automatically load the dataset, preprocess it, train the three classifiers, and output evaluation metrics alongside individual test predictions.

4. **Libraries and Functions Used and Why**
   - **Pandas & NumPy:** Used for data loading, cleaning, manipulation, and One-Hot Encoding (`pd.read_csv`, `pd.get_dummies`).
   - **Scikit-Learn (`sklearn`):**
     - `train_test_split`: To divide data into training and testing subsets.
     - `StandardScaler`: To normalize features to a standard scale.
     - `LogisticRegression`, `GaussianNB`, `KNeighborsClassifier`: The three classification algorithms trained and evaluated.
     - `accuracy_score`, `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`: For detailed metrics (Accuracy, Precision, Recall, F1-Score).
   - **Matplotlib & Seaborn:** Used for visual representation of confusion matrices and evaluation charts.

5. **Output Examples on Each Step**
   - **Preprocessing:** Successfully transformed text categories into numeric features without missing data.
   - **Model Evaluation:** Generated confusion matrices and classification reports highlighting true positives, true negatives, precision, and recall.
   - **Individual Predictions:** Verified model performance using `model.predict()` on specific feature sets, obtaining accurate results for both rejected (`0`) and accepted (`1`) candidates.

6. **Results and Model Evaluation**
   - **Logistic Regression:** Provided reliable linear classification with balanced class weighting to address target distribution.
   - **Naive Bayes:** Delivered fast, efficient probabilistic predictions based on feature independence assumptions.
   - **KNN:** Accurately classified candidate instances based on calculated feature space distances after scaling.

7. **Accuracy & Final Comparison**
   - All three models achieved high and competitive accuracy scores.
   - Models were thoroughly compared side-by-side using `classification_report` and `confusion_matrix` from `sklearn.metrics`, ensuring a well-documented machine learning project suitable for submission.