# Automated Resume Screening Application
Welcome to our Automated Resume Screening Application! 🚀 This powerful tool is designed to revolutionize the way recruiters handle resumes, making the screening process more efficient, objective, and hassle-free. Let's dive into the details of our project and how you can use it to streamline your hiring process.

![resume-screening-software](https://github.com/tuanng1102/resume-screening-app/assets/147653892/dd0bd95e-40fe-4372-a62d-7cb30d2fde76)

## Introduction

Our goal is to create an intelligent system capable of classifying resumes based on their content. By leveraging cutting-edge Natural Language Processing (NLP) techniques and the robust K-Nearest Neighbors (KNN) algorithm, we empower recruiters like you to quickly and accurately categorize resumes into relevant job categories. This not only saves time but also ensures a consistent and fair screening process for all applicants.

## Key Dependencies

Our project relies on several essential libraries and tools:

- ```pandas```: For efficient data manipulation.
- ```numpy```: For performing complex numerical computations.
- ```matplotlib.pyplot```: For creating visually appealing data visualizations.
- ```seaborn```: For advanced data visualization techniques.
- Regular Expressions ```re```: For text cleaning and preprocessing.
- scikit-learn ```sklearn```: For implementing machine learning algorithms and tasks.

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
Loads the resume dataset stored in a CSV file named "UpdatedResumeDataSet.csv" into a Pandas DataFrame named ```df```

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
Encoding the "Category" column using label encoding to convert categorical data into numerical format.

``` bash
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
```

### 6. Vectorizing text data:
The cleaned resume text is transformed into numerical features suitable for machine learning. TF-IDF (Term Frequency-Inverse Document Frequency) is employed to represent each resume as a vector, capturing the importance of words based on their frequency within the document and their overall infrequency across the entire dataset.

``` bash
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
text_vectorized = tfidf.fit_transform(df["Resume"])
```

### 7. Splitting data to train-set and test-set:
The data is divided into training and testing sets. The training set is used to train the KNN model, while the testing set is used to evaluate its performance on unseen data.


``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_vectorized, df.Category, test_size=0.2, random_state=42)
```

### 8. Training K-Nearest Neighbors model:
The KNN algorithm is chosen for its simplicity and effectiveness in text classification tasks. It classifies new resumes by finding the k nearest neighbors (most similar resumes) in the training data based on their TF-IDF vectors and assigning the most frequent category among those neighbors.

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

## Create an application with streamlit
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

![resume-desktop](https://github.com/tuanng1102/resume-screening-app/assets/147653892/5a391c48-e51d-4cd5-adc6-f6b8ac9b49e3)

### 5. Make a prediction
Upload a resume for prediction

![resume-predict](https://github.com/tuanng1102/resume-screening-app/assets/147653892/c9ade54e-4c9d-4891-b30c-b585cd5454c1)

With our Automated Resume Screening Application, recruiters can now efficiently categorize and evaluate resumes, freeing up valuable time for more strategic tasks in the hiring process. This innovative tool represents a significant step forward in modern recruitment practices, enhancing efficiency, accuracy, and overall effectiveness.
