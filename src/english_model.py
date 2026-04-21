import joblib
from nltk.corpus import stopwords
import re
import logging
import os

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("English_Model")
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


class EnglishModel:
    def __init__(self,model_e,vectorizer_e):
        self.model=joblib.load(model_e)
        self.vectorizer=joblib.load(vectorizer_e)
        self.stop=set(stopwords.words("english"))
        logger.info("English model and vectorizer initialized successfully")

    def clean_text(self,text):
        text= text.lower()
        text= re.sub(r'[^\w\s]', '', text)
        words=text.split()
        words=[word for word in words if word not in self.stop]
        return " ".join(words)
    
    def predict(self,text):
        text=self.clean_text(text)
        logger.info("English text cleaned successfully")
        text_vec=self.vectorizer.transform([text]).toarray()
        prediction=self.model.predict(text_vec)[0]
        logger.info("English prediction made successfully")
        return prediction
