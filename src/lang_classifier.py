import pandas as pd 
import joblib

class LanguageClassifier:
    def __init__(self, lang,vectorizer):
        self.lang_classifier=joblib.load(lang)
        self.vectorizer=joblib.load(vectorizer)
    
    def predict(self,text):
        text=self.vectorizer.transform([text]).toarray()
        prediction=self.lang_classifier.predict(text)[0]
        return prediction
        