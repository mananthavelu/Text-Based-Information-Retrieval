#Tokenization of Question
def stop_words(strings_list):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    bow_query_sw=([i for i in strings_list if i not in stop])
    return bow_query_sw