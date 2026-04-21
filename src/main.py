from lang_classifier import LanguageClassifier
from arabic_model import arabic_model
from english_model import EnglishModel
from config import Config
import logging
import os 

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Main")
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


lang=LanguageClassifier(Config.lang_model,Config.lang_vect)
arabic=arabic_model(Config.arabic_encod,Config.arabic_model,Config.arabic_vect)
english=EnglishModel(Config.english_model,Config.english_vect)
logger.info("All models initialized successfully")

text=input("please enter your comment to classfier it ")
logger.info("User entered text: %s", text)

language=lang.predict(text)
if language == 0:
    print("Text is arabic")
    prediction=arabic.predict(text)
    print("Prediction in is:",prediction)

else:
    print("Text is English")
    prediction=english.predict(text)
    print("Prediction in is:",prediction)



