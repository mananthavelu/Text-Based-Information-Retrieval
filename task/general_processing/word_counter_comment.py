#Creating the unique words list from the above list.
#Creating a dictionary with word counter
def word_counter_comment(strings):
    my_counter = {}
    for word in strings:
        my_counter[word] = my_counter.get(word, 0) + 1
    return my_counter