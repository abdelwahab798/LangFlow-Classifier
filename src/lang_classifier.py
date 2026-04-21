import pandas as pd 
import joblib
import logging
import os

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Lang_Model")
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

class LanguageClassifier:
    def __init__(self, lang,vectorizer):
        self.lang_classifier=joblib.load(lang)
        self.vectorizer=joblib.load(vectorizer)
        logger.info("Language classifier and vectorizer initialized successfully")

    
    def predict(self,text):
        text=self.vectorizer.transform([text]).toarray()
        prediction=self.lang_classifier.predict(text)[0]
        logger.info("Language prediction made successfully")
        return prediction
        