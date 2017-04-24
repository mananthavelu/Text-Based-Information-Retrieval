#Extracting only the words (which corresponds to the dictionary keys)
def unique_words(dictionary):    
    comments_words=[]
    for item in dictionary.keys():
        comments_words.append(item)
    return comments_words