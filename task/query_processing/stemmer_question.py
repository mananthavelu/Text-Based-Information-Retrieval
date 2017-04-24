#Stemming of the Question terms
from nltk.stem import PorterStemmer
def stemmer(strings_list):
    
    query_words=[]
    ps = PorterStemmer()
    for w in strings_list:
        query_words.append(ps.stem(w))
    return query_words