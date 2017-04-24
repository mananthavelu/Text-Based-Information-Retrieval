#Initialization with Zeros
def tf_raw_query(query_strings,wordset):    
    worddict_query=dict.fromkeys(wordset,0)    
    for word in query_strings:
        worddict_query[word]+=1
    return worddict_query