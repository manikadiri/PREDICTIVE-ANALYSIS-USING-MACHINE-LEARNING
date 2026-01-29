

Diabetes Prediction Using Logistic Regression (Python)

## Project Overview

This project implements a **machine learning model to predict diabetes** using **Logistic Regression**.
It uses a structured healthcare dataset and follows the complete ML pipeline including **data preprocessing, feature scaling, model training, and evaluation**.

The aim is to classify whether a patient is **diabetic or non-diabetic** based on medical attributes.

---

##  Technologies & Libraries Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and evaluation

  * `train_test_split`
  * `StandardScaler`
  * `LogisticRegression`
  * `accuracy_score`
  * `confusion_matrix`

---

##  Dataset Information

* **File name:** `diabetes.csv`
* **Format:** CSV
* **Target Column:** `Outcome`

  * `0` → Non-diabetic
  * `1` → Diabetic
* **Features:** Medical attributes such as glucose level, BMI, blood pressure, etc.

---

## ⚙️ Workflow Explanation (Code Analysis)

### 1️ Import Required Libraries

The script imports libraries for:

* Data handling (`pandas`, `numpy`)
* Machine learning model building
* Model evaluation

---

### 2️⃣ Load Dataset

```python
pd.read_csv("diabetes.csv")
```

Loads the diabetes dataset from the local system into a Pandas DataFrame.

---

### 3️⃣ Data Inspection

* Displays dataset shape (rows × columns)
* Shows the first few records using `head()`

This helps in understanding the dataset structure.

---

### 4️⃣ Handle Missing Values

```python
data.fillna(data.mean(), inplace=True)
```

* Replaces missing values with the **mean of each column**
* Ensures the dataset is clean before training

---

### 5️⃣ Feature & Target Separation

```python
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
```

* `X` → Input features
* `y` → Target label (diabetes outcome)

---

### 6️⃣ Train-Test Split

```python
train_test_split(test_size=0.2, random_state=42)
```

* 80% data for training
* 20% data for testing
* Ensures reproducibility using a fixed random seed

---

### 7️⃣ Feature Scaling

```python
StandardScaler()
```

* Standardizes features to mean = 0 and standard deviation = 1
* Improves Logistic Regression performance

---

### 8️⃣ Model Training

```python
LogisticRegression().fit(X_train, y_train)
```

* Trains a Logistic Regression classifier
* Learns the relationship between features and diabetes outcome

---

### 9️⃣ Model Prediction

```python
model.predict(X_test)
```

* Predicts diabetes outcomes on unseen test data

---

### 🔟 Model Evaluation

```python
accuracy_score()
confusion_matrix()
```

* **Accuracy Score** → Measures overall correctness
* **Confusion Matrix** → Shows:

  * True Positives
  * True Negatives
  * False Positives
  * False Negatives

---

##  Output Generated

The program prints:

* Dataset shape and preview
* Model accuracy
* Confusion matrix

These outputs help assess model performance.

---

## ▶️ How to Run the Project

### Prerequisites

* Python 3.x
* Required libraries installed:

```bash
pip install pandas numpy scikit-learn
```

### Run the Script

```bash
python mani.py
```

---

## 🚀 Use Cases

* Healthcare data analysis
* Diabetes risk prediction
* Machine learning practice project
* Academic mini / major project

---

