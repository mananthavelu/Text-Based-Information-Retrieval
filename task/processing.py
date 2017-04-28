import math
from nltk.stem import PorterStemmer

def log_wt(val1):
    if val1!=0:
        value=1+(math.log(val1))
    else:
        value=0
    return value