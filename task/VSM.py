{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Tokenization of Question\n",
    "def tokenization(list_of_strings):\n",
    "    BOW_Query=list_of_strings.split()\n",
    "    return BOW_Query"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Removal of Stop words in the Question\n",
    "def stop_words(strings_list):\n",
    "    from nltk.corpus import stopwords\n",
    "    stop = set(stopwords.words('english'))\n",
    "    bow_query_sw=([i for i in strings_list if i not in stop])\n",
    "    return bow_query_sw"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Stemming of the Question terms\n",
    "def stemmer(strings_list):\n",
    "    from nltk.stem import PorterStemmer\n",
    "    query_words=[]\n",
    "    ps = PorterStemmer()\n",
    "    for w in strings_list:\n",
    "        query_words.append(ps.stem(w))\n",
    "    return query_words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Removal of stop words in the answers\n",
    "def sw_co(comments):\n",
    "    from nltk.corpus import stopwords\n",
    "    stop = set(stopwords.words('english'))\n",
    "    comments_sw=[]\n",
    "    for item in comments:\n",
    "        stop_words=([i for i in item.lower().split() if i not in stop])\n",
    "        comments_sw.append(stop_words)\n",
    "    return comments_sw"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Stemming the words in the comments\n",
    "def stemmer_co(strings):\n",
    "    from nltk.stem import PorterStemmer\n",
    "    comments_st=[]\n",
    "    ps = PorterStemmer()\n",
    "    for items in strings:\n",
    "        comment_st_each=[]\n",
    "        for w in items:\n",
    "            comment_st_each.append(ps.stem(w))\n",
    "        comments_st.append(comment_st_each)\n",
    "    return comments_st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Creating the unique words list from the above list.\n",
    "#Creating a dictionary with word counter\n",
    "def word_counter(strings):    \n",
    "    my_counter = {}\n",
    "    for line in strings:\n",
    "        for word in line:\n",
    "            my_counter[word] = my_counter.get(word, 0) + 1\n",
    "    return my_counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Extracting only the words (which corresponds to the dictionary keys)\n",
    "def unique_words(dictionary):    \n",
    "    comments_words=[]\n",
    "    for item in dictionary.keys():\n",
    "        comments_words.append(item)\n",
    "    return comments_words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def wordset(querydoc,commentsdoc):\n",
    "    wordset=set(querydoc).union(set(commentsdoc))\n",
    "    return wordset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Initialization with Zeros\n",
    "def tf_raw_query(query_strings,wordset):    \n",
    "    worddict_query=dict.fromkeys(wordset,0)    \n",
    "    for word in query_strings:\n",
    "        worddict_query[word]+=1\n",
    "    return worddict_query"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Calculation of the tf-weight terms\n",
    "def tf_weight(words_raw):    \n",
    "    dict_tf_wt={}\n",
    "    import math\n",
    "    for key,val in words_raw.items():\n",
    "        if val!=0:\n",
    "            value=1+(math.log(val))\n",
    "        else:\n",
    "            value=0\n",
    "        dict_tf_wt[key]=value\n",
    "    return dict_tf_wt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 330,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Find the documents frequency - Number of documents a term exist\n",
    "def df_df(wordset,doc):   \n",
    "    dict_df={}\n",
    "    for item in wordset:\n",
    "        dfg=0\n",
    "        for item2 in doc:\n",
    "            if item in item2:\n",
    "                dfg+=1            \n",
    "            else:\n",
    "                dfg=0\n",
    "            dict_df[item]=dfg\n",
    "    return dict_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Calculating the idf for a query\n",
    "def idf_query(df):\n",
    "    idf=[]\n",
    "    import math\n",
    "    for item in df['df']:\n",
    "        if item !=0:\n",
    "            value=math.log(10/item)\n",
    "        else:\n",
    "            value=0\n",
    "        idf.append(value)\n",
    "    df[\"idf\"]=idf\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Normalizing the weights\n",
    "def norm_q(df):    \n",
    "    sum1=0.0\n",
    "    for item in df['wt']:\n",
    "        sq=item*item\n",
    "        sum1+=sq\n",
    "    import math\n",
    "    norm_q=[]\n",
    "    for itemm in df['wt']:\n",
    "        if item==0:\n",
    "            deno=0\n",
    "        else:\n",
    "            deno=itemm/math.sqrt((sum1))\n",
    "        norm_q.append(deno)\n",
    "    df[\"nor_q\"]=norm_q\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def log_wt(val):\n",
    "    import math\n",
    "    if val!=0:\n",
    "        value=1+(math.log(val))\n",
    "    else:\n",
    "        value=0\n",
    "    return val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Importing the Question-Comments XML file,Identifying the Question-Comments within the XML file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Importing the XML file and required libraries\n",
    "import xml.etree.ElementTree as ET\n",
    "import pandas as pd\n",
    "tree = ET.parse('small_train_1.xml')\n",
    "root = tree.getroot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Jessica\\Miniconda3\\envs\\py35\\lib\\site-packages\\ipykernel\\__main__.py:93: RuntimeWarning: divide by zero encountered in true_divide\n"
     ]
    }
   ],
   "source": [
    "#Extracting the Questions and comments\n",
    "#Question\n",
    "for thread in root.iter(\"Thread\"):\n",
    "    for question in root.iter('RelQuestion'):\n",
    "        qa=question.attrib['RELQ_ID']\n",
    "        query=\"\"\n",
    "        for question in root.iter('RelQBody'):\n",
    "            each_question=question.text\n",
    "            query+=each_question\n",
    "\n",
    "            #Tokenization of the Question\n",
    "            hey=tokenization(query)\n",
    "\n",
    "            #Removing the stop words from the Question\n",
    "            sq=stop_words(hey)\n",
    "\n",
    "            #Stemming the words in Question\n",
    "            st=stemmer(sq)\n",
    "\n",
    "            #Comments\n",
    "            comments=[]\n",
    "            ids=[]\n",
    "        for each_answer in root.iter('RelCText'):\n",
    "            answer=each_answer.text\n",
    "            comments.append(answer)\n",
    "            for comment in root.iter('RelComment'):\n",
    "                relc_id=comment.attrib['RELC_ID']\n",
    "                ids.append(relc_id)\n",
    "\n",
    "        result=pd.DataFrame()\n",
    "            #Removing stop words from the Comments    \n",
    "        sc_co=sw_co(comments)\n",
    "\n",
    "            #Stemming the words in the comments\n",
    "        st_co=stemmer_co(sc_co)\n",
    "\n",
    "            #Creating a wordset for the comments\n",
    "        counter1=word_counter(st_co)\n",
    "        unique_words1=unique_words(counter1)\n",
    "        wordset1=wordset(st,unique_words1)\n",
    "\n",
    "            #Calculating the raw terms in the comments\n",
    "        tf_raw=tf_raw_query(st,wordset1)\n",
    "        tf_wt=tf_weight(tf_raw)\n",
    "        df_query=df_df(wordset1,st_co)\n",
    "\n",
    "            #Appending the tf-raw, tf-wt, df into a dataframe\n",
    "\n",
    "        dframe=pd.DataFrame.from_dict([tf_raw,tf_wt,df_query],orient='columns')\n",
    "        df=dframe.transpose()\n",
    "        summary_of_results=df.rename(columns={0: 'tf_raw',1: 'tf-weight',2: 'df'})\n",
    "\n",
    "            #Calculating the IDF for the query\n",
    "        idf_query=idf_query(summary_of_results)\n",
    "        idf_query[\"wt\"]=idf_query.idf*idf_query.tf_raw\n",
    "\n",
    "            #Normalization\n",
    "        norm_query=norm_q(idf_query)\n",
    "\n",
    "            #Calculating the raw-terms for all the comments\n",
    "\n",
    "        for item in st_co:\n",
    "            worddict_terms=dict.fromkeys(wordset1,0)  \n",
    "            frames=pd.DataFrame()\n",
    "            for items in item:\n",
    "                worddict_terms[items]+=1\n",
    "                df_com_c1=pd.DataFrame.from_dict([worddict_terms])\n",
    "            frames=[result,df_com_c1]\n",
    "            result = pd.concat(frames)\n",
    "        hey=result.transpose()\n",
    "\n",
    "            #Calculating the terms weight\n",
    "        hey_1=hey.applymap(log_wt)\n",
    "\n",
    "            #Adding the column name to the Dataframe\n",
    "        hey_1.columns=[\"tf_doc_wt_1\",\"tf_doc_wt_2\",\"tf_doc_wt_3\",\"tf_doc_wt_4\",\"tf_doc_wt_5\",\"tf_doc_wt_6\",\"tf_doc_wt_7\",\"tf_doc_wt_8\",\"tf_doc_wt_9\",\n",
    "                           \"tf_doc_wt_10\"]\n",
    "\n",
    "\n",
    "            #Normalization of all the comments weights\n",
    "        counter=1\n",
    "        for item in hey_1:\n",
    "            sum2=0.0\n",
    "            for items in hey_1[item]:\n",
    "                sq1=items*items\n",
    "                sum2+=sq1\n",
    "                import math\n",
    "                norm_q_d=[]\n",
    "                for itemw in hey_1[item]:\n",
    "                    if itemw==0:\n",
    "                        deno=0\n",
    "                    else:\n",
    "                        deno=itemw/math.sqrt((sum2))\n",
    "                    norm_q_d.append(deno)\n",
    "            s=str(counter)\n",
    "            hey_1[s]=norm_q_d\n",
    "            counter+=1\n",
    "\n",
    "            #Keeping the column weights \n",
    "        cols = [0,1,2,3,4,5,6,7,8,9]\n",
    "        hey_1.drop(hey_1.columns[cols],axis=1,inplace=True)\n",
    "\n",
    "            #product\n",
    "        columns = hey_1.columns[:]    \n",
    "        for item in columns:\n",
    "            if item !=0:\n",
    "                hey_1[columns] *= norm_query['nor_q']\n",
    "            else:\n",
    "                hey_1[columns]=0\n",
    "\n",
    "            #Replacing the NaN values with Zero's\n",
    "        hey_2=hey_1.fillna('0')\n",
    "\n",
    "            #Query - Documents score\n",
    "        Cosine_Score=hey_2.sum(axis=0)       \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1     0.0\n",
       "2     0.0\n",
       "3     0.0\n",
       "4     0.0\n",
       "5     0.0\n",
       "6     0.0\n",
       "7     0.0\n",
       "8     0.0\n",
       "9     0.0\n",
       "10    0.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 336,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Cosine_Score"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
