import joblib
from nltk.corpus import stopwords
import re

class EnglishModel:
    def __init__(self,model_e,vectorizer_e):
        self.model=joblib.load(model_e)
        self.vectorizer=joblib.load(vectorizer_e)
        self.stop=set(stopwords.words("english"))

    def clean_text(self,text):
        text= text.lower()
        text= re.sub(r'[^\w\s]', '', text)
        words=text.split()
        words=[word for word in words if word not in self.stop]
        return " ".join(words)
    
    def predict(self,text):
        text=self.clean_text(text)
        text_vec=self.vectorizer.transform([text]).toarray()
        prediction=self.model.predict(text_vec)[0]
        return prediction
