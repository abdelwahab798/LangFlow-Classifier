import re
import nltk
import joblib
import logging
import os

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Arabic_Model")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"Project.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)


class arabic_model():
    def __init__(self,labler_encoder_a,model_a,vectorizer_a):
        self.model=joblib.load(model_a)
        self.labler=joblib.load(labler_encoder_a)
        self.vectorizer_a=joblib.load(vectorizer_a)
        self.stop=nltk.corpus.stopwords.words("arabic")
        logger.info("arabic model and vectorizer initialized successfully")

    def clean_text(self,text):
        text=re.sub("[إأآا]", "ا", text)
        text = re.sub(r'[^\u0600-\u06FF\s]','', text)
        text=re.sub("ى", "ي", text)
        words=text.split()
        words=[word for word in words if word not in self.stop]
        return " ".join(words)
    
    def predict(self,text):
        text=self.clean_text(text)
        logger.info("Arabic text cleaned successfully")
        vecotize_text=self.vectorizer_a.transform([text]).toarray()
        prediction=self.model.predict(vecotize_text)[0]
        logger.info("Arabic prediction made successfully")
        return self.labler.inverse_transform([prediction])[0]


        