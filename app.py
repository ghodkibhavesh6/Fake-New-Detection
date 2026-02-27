import streamlit as st
import pickle

#load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection")

st.write("Enter news text to check whether it is Real or Fake")

news_text = st.text_area("News Content")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([news_text])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")