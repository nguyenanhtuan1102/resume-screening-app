# Introduction

This project utilizes logistic regression and Streamlit to automate resume screening, empowering recruiters to efficiently evaluate job applicants' qualifications. By combining machine learning with a user-friendly interface, this app streamlines the recruitment process, saving time and ensuring fair evaluation.

![resume-screening-software](https://github.com/tuanng1102/resume-screening-app/assets/147653892/dd0bd95e-40fe-4372-a62d-7cb30d2fde76)

# Dependencies

The project relies on several Python libraries, including ```numpy```, ```pandas```, ```matplotlib```, ```seaborn```, and Regular Expression ```re```, to facilitate data manipulation, visualization, and text cleaning processes.

# Explanation of the Code:

## 1. Data Preprocessing
### 1.1 Import libraries:
Importing necessary libraries for data manipulation, visualization, and natural language processing.

``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
```

### 1.2 Load dataset:
Loads the resume dataset stored in a CSV file named "UpdatedResumeDataSet.csv" into a Pandas DataFrame named ```df```

``` bash
df = pd.read_csv("UpdatedResumeDataSet.csv")
```

### 1.3 Exploring dataset:
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

### 1.4 Cleaning data:
There are 4 things that I must solve:
- URLs
- Hashtags
- Mentions
- Special letter
- Punctuation

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

### 1.5 Encoding categorical data:
Encoding the "Category" column using label encoding to convert categorical data into numerical format.

``` bash
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
```

### 1.6 Vectorizing text data:
Transforming the text data in the "Resume" column into numerical vectors using TF-IDF vectorization.

``` bash
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
text_vectorized = tfidf.fit_transform(df["Resume"])
```

### 1.7 Splitting data to train-set and test-set:
Splitting the dataset into training and testing sets for model training and evaluation.

``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_vectorized, df.Category, test_size=0.2, random_state=42)
```

## 2. Modelling
### 2.1 Training K-Nearest Neighbors model:
Training a K-Nearest Neighbors classifier using the training data.

``` bash
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = OneVsRestClassifier(KNeighborsClassifier())
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```

### 2.2 Predict model:
Making predictions on the test data using the trained classifier.

``` bash
y_pred = clf.predict(X_test)
```

### 2.3 Evaluate:
Evaluating the model's accuracy using the test data.

``` bash
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: ",accuracy_score(y_test, y_pred))
```

## 3. Prediction System
### 3.1 Save model

``` bash
import pickle
pickle.dump(tfidf, open('tfidf.pkl','wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```
