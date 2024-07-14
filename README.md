# Automated Resume Screening Application
Welcome to our Automated Resume Screening Application! 🚀 This powerful tool is designed to revolutionize the way recruiters handle resumes, making the screening process more efficient, objective, and hassle-free. Let's dive into the details of our project and how you can use it to streamline your hiring process.

![resume](https://github.com/user-attachments/assets/9ca62562-d063-43ad-9ba2-91fae9b7946d)

## Introduction

Our goal is to create an intelligent system capable of classifying resumes based on their content. By leveraging cutting-edge Natural Language Processing (NLP) techniques and the robust K-Nearest Neighbors (KNN) algorithm, we empower recruiters like you to quickly and accurately categorize resumes into relevant job categories. This not only saves time but also ensures a consistent and fair screening process for all applicants.

## Key Dependencies

Our project relies on several essential libraries and tools:

- ```pandas```: This library is indispensable for data manipulation and analysis, providing powerful data structures and tools for working with structured data.
- ```numpy```: NumPy is the fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- ```matplotlib.pyplot```: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. We specifically import the pyplot module for creating plots and charts.
- ```seaborn```: Seaborn is a statistical data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- ```re```: The re module provides support for working with regular expressions in Python, which are powerful tools for pattern matching and text manipulation.
- ```sklearn```: Scikit-learn is a machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It includes a wide range of machine learning algorithms and utilities for preprocessing, model selection, and evaluation.


## Model Preparation:

Let's walk through the steps we took to prepare our model:

### 1. Import libraries:
We start by importing the necessary libraries for data manipulation, visualization, and NLP.

``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
```

### 2. Load dataset:
This section involves loading the resume dataset from a CSV file into a Pandas DataFrame named ```df```. The dataset contains information about job applicants, including their resumes and corresponding job categories.

``` bash
df = pd.read_csv("UpdatedResumeDataSet.csv")
```

### 3. Exploring dataset:
This step is crucial for understanding the characteristics of the dataset and identifying potential issues. Here, we perform value counts and view unique values in the "Category" column to gain insights into the distribution of job categories. Furthermore, we create a bar plot to visualize the distribution of different job categories visually.

``` bash
# Value counts
df["Category"].value_counts()

# Unique values of Category column
df.Category.unique()

# Set categories and values for barplot
categories = list(df.Category.unique())
values = list(df.Category.value_counts().values)

# visualize bar plot
plt.figure(figsize = (12, 7))
plt.barh(categories, values, color="red", height=0.7)
plt.xticks(rotation=90)
plt.show()
```

![1](https://github.com/tuanng1102/resume-screening-app/assets/147653892/2e558a00-d5b8-4faf-8510-042f67b16aa1)

### 4. Cleaning data:
A crucial step involves cleaning the resume text to remove irrelevant information and prepare it for machine learning. This includes removing:
- URLs (e.g., http://www.example.com)
- Hashtags (e.g., #datascience)
- Mentions (e.g., @johnDoe)
- Special characters and punctuation (except for alphanumeric characters and whitespace)
- Non-ASCII characters

``` bash
# Create a function that cleans text data
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Apply function
df["Resume"] = df["Resume"].apply(lambda x: cleanResume(x))
```

### 5. Encoding categorical data:
Machine learning algorithms require numerical input, so we use label encoding to convert the categorical job categories into numerical labels. The ```LabelEncoder``` from scikit-learn is used to achieve this transformation.

``` bash
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
```

### 6. Vectorizing text data:
Text data cannot be directly used for machine learning algorithms, so we need to convert it into a numerical format. We use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to transform the cleaned resume text into a matrix of TF-IDF features. This process assigns numerical values to each word in the resume based on its frequency and importance in the document and the entire dataset.

``` bash
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
text_vectorized = tfidf.fit_transform(df["Resume"])
```

### 7. Splitting data to train-set and test-set:
Before training our machine learning model, we split the dataset into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance. The ```train_test_split function``` from scikit-learn is used for this purpose.


``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_vectorized, df.Category, test_size=0.2, random_state=42)
```

### 8. Training K-Nearest Neighbors model:
In this section, we train a K-Nearest Neighbors (KNN) classifier on the training data. KNN is a simple and effective algorithm for classification tasks, particularly in text classification. We initialize the KNeighborsClassifier from scikit-learn and fit it to the training data.


``` bash
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = OneVsRestClassifier(KNeighborsClassifier())
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```

### 9. Predict model:
Making predictions on the test data using the trained classifier.

``` bash
y_pred = clf.predict(X_test)
```

### 10. Evaluate:
The trained KNN model is evaluated on the testing set using accuracy as the primary metric. Accuracy measures the proportion of correctly classified resumes. Additionally, confusion matrix visualization can provide insights into the model's performance for each category.


``` bash
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: ",accuracy_score(y_test, y_pred))
```

### 11. Save model
The trained ```TF-IDF``` vectorizer and KNN model are serialized using pickle for future use. This allows for efficient deployment of the classification system without retraining the model on every execution.


``` bash
import pickle
pickle.dump(tfidf, open('tfidf.pkl','wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```

## Create A Application
### 1. Load model
We load the serialized model and TF-IDF vectorizer.

``` bash
# loading model
clf = pickle.load(open("model/clf.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
```

### 2. Clean text
We define a function to clean the resume text.

``` bash
# clean text
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text
```

### 3. Web Application
Our Streamlit app allows users to upload a resume file and predicts its category.

The Streamlit app does the following:
- Displays a title: "Resume Screening App".
- Provides a file uploader for users to select a resume in TXT or PDF format.
- If a file is uploaded:
    - Reads the uploaded file as bytes.
    - Attempts to decode the bytes as UTF-8 (common encoding). If errors occur, falls back to Latin-1 encoding.
    - Cleans the resume text using the clean_resume function.
    - Transforms the cleaned text into numerical features using the loaded TF-IDF vectorizer (tfidf).
    - Makes a prediction on the job category using the loaded classification model (clf).
    - Displays the predicted category.
    - Maps the predicted category ID to a human-readable category name using a dictionary (category_mapping).
    - Displays the predicted category name.

``` bash
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        input_feature = tfidf.transform([cleaned_resume])
        predictions = clf.predict(input_feature)[0]
        st.write(predictions)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(predictions, "Unknown")
        st.write("Predicted Category:", category_name)


# python main
if __name__ == "__main__":
    main()
```

### 4. Running the app
To run the application:

``` bash
streamlit run resume-screening-app.py
```

![resume-desktop-1](https://github.com/tuanng1102/resume-screening-app/assets/147653892/b781fbce-c84c-4054-9ba3-aeac806f3892)

### 5. Make a prediction
Upload a resume for prediction.

![resume-predict-1](https://github.com/tuanng1102/resume-screening-app/assets/147653892/abff0ff9-bfd4-4b73-bf88-52e537c36d87)

With our Automated Resume Screening Application, recruiters can now efficiently categorize and evaluate resumes, freeing up valuable time for more strategic tasks in the hiring process. This innovative tool represents a significant step forward in modern recruitment practices, enhancing efficiency, accuracy, and overall effectiveness.
