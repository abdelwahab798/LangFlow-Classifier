from dotenv import load_dotenv
import os

load_dotenv()
class Config():
    arabic_model=os.getenv("arabic_model")
    arabic_vect=os.getenv("arabic_vect")
    arabic_encod=os.getenv("arabic_encod")
    english_model=os.getenv("english_model")
    english_vect=os.getenv("english_vect")
    lang_model=os.getenv("lang_model")
    lang_vect=os.getenv("lang_vect")
