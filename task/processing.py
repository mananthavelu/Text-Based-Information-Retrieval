import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Calculating the weights from the raw-values
def log_wt(val1):
    if val1!=0:
        value=1+(math.log(val1))
    else:
        value=0
    return value


# Stemming of the Question terms
def stemmer(strings_list):
    query_words = []
    ps = PorterStemmer()
    for w in strings_list:
        query_words.append(ps.stem(w))
    return query_words


# Tokenization of Question
def stop_words(strings_list):
    stop = set(stopwords.words('english'))
    bow_query_sw=([i for i in strings_list if i not in stop])
    return bow_query_sw


# Tokenization of Question
def tokenization(list_of_strings):
    BOW_Query=list_of_strings.split()
    return BOW_Query


# Normalizing the weights
def norm_weight(weights):
    sum1=0.0
    for item in weights:
        sq=item*item
        sum1+=sq
    norm_weights=[]
    for itemm in weights:
        if itemm==0:
            deno=0
        else:
            deno=itemm/(math.sqrt((sum1)))
        norm_weights.append(deno)
    return norm_weights


# Calculating the idf for a query
def calculate_idf_query(dfw):
    idf=[]
    for item in dfw['df']:
        if item !=0:
            value=math.log(10/item)
        else:
            value=0
        idf.append(value)
    dfw["idf"]=idf
    return dfw

# Calculating the document frequcny for a term
def calculate_document_frequency(ww,dd):
    dict_dfdf={}
    for item in ww:
        co=0
        for items in dd:
            if item in items:
                co+=1
            else:
                pass
            dict_dfdf[item]=co
     return dict_dfdf


# Initialization with Zeros
def tf_raw_query(query_strings,wordset):
    worddict_query=dict.fromkeys(wordset,0)
    for word in query_strings:
        worddict_query[word]+=1
    return worddict_query

# Calculation of the tf-weight terms
def tf_weight(words_raw):
    dict_tf_wt={}
    import math
    for key,val in words_raw.items():
        if val!=0:
            value=1+(math.log(val))
        else:
            value=0
        dict_tf_wt[key]=value
    return dict_tf_wt