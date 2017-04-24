#Stemming the words in the comments

def stemmer_co(strings):
	from nltk.stem import PorterStemmer
	comments_st=[]
	ps = PorterStemmer()
	for items in strings:
		comment_st_each=[]
		for w in items:
			comment_st_each.append(ps.stem(w))
		comments_st.append(comment_st_each)
	return comments_st