# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:54:57 2026

@author: shrey
"""

import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv("fake_or_real_news.csv")

X = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf.fit_transform(x_train)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        vector = tfidf.transform([news])

        result = model.predict(vector)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)