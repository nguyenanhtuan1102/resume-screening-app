# Introduction

This project presents a machine learning approach for classifying resumes based on their content. It leverages Natural Language Processing (NLP) techniques for text cleaning, feature extraction, and classification using the K-Nearest Neighbors (KNN) algorithm. The system aims to assist recruiters in efficiently screening resumes by automatically assigning them to relevant categories.

![resume-screening-software](https://github.com/tuanng1102/resume-screening-app/assets/147653892/dd0bd95e-40fe-4372-a62d-7cb30d2fde76)

# Dependencies

- ```pandas``` (for data manipulation)
- ```numpy``` (for numerical computations)
- ```matplotlib.pyplot``` (for data visualization)
- ```seaborn``` (for advanced data visualization)
- Regular Expressions ```re``` (for text cleaning)
- ```sklearn``` (for machine learning tasks)


# Explanation of the Code:

### 1. Import libraries:
Importing necessary libraries for data manipulation, visualization, and natural language processing.

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
The trained TF-IDF vectorizer and KNN model are serialized using pickle for future use. This allows for efficient deployment of the classification system without retraining the model on every execution.


``` bash
import pickle
pickle.dump(tfidf, open('tfidf.pkl','wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```
