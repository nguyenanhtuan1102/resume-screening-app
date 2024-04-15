import streamlit as st
import pickle
import nltk
import re

nltk.download("punkt")
nltk.download("stopwords")

# loading model
clf = pickle.load(open("model/clf.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

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

# web app
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