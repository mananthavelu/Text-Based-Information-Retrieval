# Creating the unique words list from the above list.
# Creating a dictionary with word counter

#Word counter
def word_counter(strings):
    my_counter = {}
    for line in strings:
        for word in line:
            my_counter[word] = my_counter.get(word, 0) + 1
    return my_counter


# Extracting only the words (which corresponds to the dictionary keys)
def unique_words(dictionary):
    comments_words=[]
    for item in dictionary.keys():
        comments_words.append(item)
    return comments_words


#Creating a wordset for a question and all the comments in a thread
def wordset(querydoc,commentsdoc):
    wordset=set(querydoc).union(set(commentsdoc))
    return wordset