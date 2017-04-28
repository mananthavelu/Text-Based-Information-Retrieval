#Removal of stop words in the answers

def sw_co(comments):
	from nltk.corpus import stopwords
	stop = set(stopwords.words('english'))
	comments_sw=[]
	for item in comments:
		stop_words=([i.lower() for i in item if i not in stop])
		comments_sw.append(stop_words)
	return comments_sw