import re
import nltk
import joblib


class arabic_model():
    def __init__(self,labler_encoder_a,model_a,vectorizer_a):
        self.model=joblib.load(model_a)
        self.labler=joblib.load(labler_encoder_a)
        self.vectorizer_a=joblib.load(vectorizer_a)
        self.stop=nltk.corpus.stopwords.words("arabic")

    def clean_text(self,text):
        text=re.sub("[إأآا]", "ا", text)
        text = re.sub(r'[^\u0600-\u06FF\s]','', text)
        text=re.sub("ى", "ي", text)
        words=text.split()
        words=[word for word in words if word not in self.stop]
        return " ".join(words)
    
    def predict(self,text):
        text=self.clean_text(text)
        vecotize_text=self.vectorizer_a.transform([text]).toarray()
        prediction=self.model.predict(vecotize_text)[0]
        return self.labler.inverse_transform([prediction])[0]


        