def wordset(querydoc,commentsdoc):
    wordset=set(querydoc).union(set(commentsdoc))
    return wordset