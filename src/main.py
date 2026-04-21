from lang_classifier import LanguageClassifier
from arabic_model import arabic_model
from english_model import EnglishModel
from config import Config

lang=LanguageClassifier(Config.lang_model,Config.lang_vect)
arabic=arabic_model(Config.arabic_encod,Config.arabic_model,Config.arabic_vect)
english=EnglishModel(Config.english_model,Config.english_vect)

text=input("please enter your comment to classfier it ")

language=lang.predict(text)
if language == 0:
    print("Text is arabic")
    prediction=arabic.predict(text)
    print("Prediction in is:",prediction)
else:
    print("Text is English")
    prediction=english.predict(text)
    print("Prediction in is:",prediction)



