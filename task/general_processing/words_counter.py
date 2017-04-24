#Creating the unique words list from the above list.
#Creating a dictionary with word counter
def word_counter(strings):    
    my_counter = {}
    for line in strings:
        for word in line:
            my_counter[word] = my_counter.get(word, 0) + 1
    return my_counter