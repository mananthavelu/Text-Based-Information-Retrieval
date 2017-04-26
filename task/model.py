#Importing the XML file and required libraries
import xml.etree.ElementTree as ET
import pandas as pd
import datetime
from nltk.corpus import stopwords

import query_processing
from query_processing import tokenization_question
from query_processing import document_frequency
from query_processing import idf_query_terms
from query_processing import normalization_query
from query_processing import stemmer_question
from query_processing import stop_words_question
from query_processing import tf_raw_query_terms
from query_processing import tf_weight_terms

import comments_processing
from comments_processing import stemmer_comments
from comments_processing import stop_words_comments
from comments_processing import log_weight

import general_processing
from general_processing import unique_words_list
from general_processing import words_counter
from general_processing import wordsets



tree = ET.parse('SemEval2016-Task3-CQA-QL-dev-subtaskA.xml')
root = tree.getroot()

#Initializing the scores
scores={}

#Execution Start time
start_exe=datetime.datetime.now()

#Iterating over the Threads for cosine scoring

for thread in root.findall("Thread"):
    start = datetime.datetime.now()
    dict_score={}
    list_result=[]
    
    #Accessing the Question in a Thread
    
    for question in thread.findall('RelQuestion'):
        Question_ID=question.attrib['RELQ_ID']
        Question_text_retrieved=question.find('RelQBody').text  
        
        

        #Removing the punctuation in the Question
        
        import string
        translate_table = dict((ord(char), None) for char in string.punctuation)

        #Removing the empty questions

        if not Question_text_retrieved:
            continue

        Question_text=Question_text_retrieved.translate(translate_table)
        
        
            
        print ("Accessing the Question ID ",Question_ID)

        #Tokenization of the Question
        Tokenized_Question=tokenization_question.tokenization(Question_text)
        
        #Removing the stop words from the Question        
        Question_Stop_words=stop_words_question.stop_words(Tokenized_Question)        
        
        #Stemming the words in Question        
        Stemmed_question=stemmer_question.stemmer(Question_Stop_words)
    
    if not Question_text:
            print ('skipping',Question_ID)
            continue
        
    #Comments    
    print ("Accessing the comments for ",Question_ID)         
    comments=[]    
    comments_ids=[]
    
    for each_answer in thread.findall('RelComment'):
        Answer_ID=each_answer.attrib['RELC_ID']
        Answer_text_retrieved=each_answer.find('RelCText').text

        import string
        translate_table_2 = dict((ord(char), None) for char in string.punctuation)
        Answer_text=Answer_text_retrieved.translate(translate_table_2)
        
        comments.append(Answer_text)
        comments_ids.append(Answer_ID)
		
    #Removing stop words from the Comments   
    Comments_stop_words=stop_words_comments.sw_co(comments)
    
    #Stemming the words in the comments   
    Comments_Stemming=stemmer_comments.stemmer_co(Comments_stop_words) 
    
    
    #Creating a wordset for the comments
    counter1=words_counter.word_counter(Comments_Stemming)
    unique_words1=unique_words_list.unique_words(counter1)
    wordset1=wordsets.wordset(Stemmed_question,unique_words1)
    
    
    #Calculating the raw terms in the comments
    tf_raw=tf_raw_query_terms.tf_raw_query(Stemmed_question,wordset1) 
    tf_wt=tf_weight_terms.tf_weight(tf_raw)
    
    dl=list(wordset1)
    df_query=document_frequency.dfdfdf(dl,Comments_Stemming)

    #Appending the tf-raw, tf-wt, df into a dataframe
    
    dframe=pd.DataFrame.from_dict([tf_raw,tf_wt,df_query],orient='columns')
    df=dframe.transpose()
    summary_of_results=df.rename(columns={0: 'tf_raw',1: 'tf_weight',2: 'df'})
    
    #Calculating the IDF for the query    
    idf_query=idf_query_terms.idf_query1(summary_of_results)    
    idf_query["wt"]=idf_query.idf*idf_query.tf_weight
    
    
    #Normalization   
    norm_query=normalization_query.norm_q(idf_query)
    
    #Calculating the raw-terms for all the comments    
    result=pd.DataFrame()
    for item in Comments_Stemming:
        worddict_terms=dict.fromkeys(wordset1,0)
        for items in item:
            worddict_terms[items]+=1
            df_com_c1=pd.DataFrame.from_dict([worddict_terms])
        frames=[result,df_com_c1]
        result = pd.concat(frames)
    Comments_raw_terms=result.transpose()
    Comments_weights=Comments_raw_terms.applymap(log_weight.log_wt)
    
    #Adding the column names to the Dataframe
    Comments_weights.columns=["tf_doc_wt_1","tf_doc_wt_2","tf_doc_wt_3","tf_doc_wt_4","tf_doc_wt_5",
                              "tf_doc_wt_6","tf_doc_wt_7","tf_doc_wt_8","tf_doc_wt_9","tf_doc_wt_10"]

    #Normalization of all the comments weights
    
    counter=1
    for item in Comments_weights:
        sum2=0.0
        for items in Comments_weights[item]:
            sq1=items*items
            sum2+=sq1
            import math
            norm_q_d=[]
            for itemw in Comments_weights[item]:
                if itemw==0:
                    deno=0
                else:
                    deno=itemw/math.sqrt((sum2))
                norm_q_d.append(deno)
        s=str(counter)
        Comments_weights[s]=norm_q_d
        counter+=1

    #Keeping the column weights 
    
    cols = [0,1,2,3,4,5,6,7,8,9]
    Comments_weights.drop(Comments_weights.columns[cols],axis=1,inplace=True)
    
    #product
    
    columns = Comments_weights.columns[:]
    for item in columns:
        Comments_weights[item] *= norm_query['nor_q']

   
    #Replacing the NaN values with Zero's
    
    Comments_weights_corrected=Comments_weights.fillna('0')

    #Converting to Float datatype - This is to avoid the empty cosine score
    Check=Comments_weights_corrected.convert_objects(convert_numeric=True)

    Cosine_Score=Check.sum(axis=0)
    
    scoreee=list(Cosine_Score)
    
    dictionary = dict(zip(comments_ids, scoreee))
    scores.update(dictionary)
 
    for item in sorted(dictionary.keys()):
        list_3=[Question_ID,item,dictionary[item]]
        list_result.append(list_3)              
    
    
    tmp=sorted(list_result,key=lambda x:float(x[2]),reverse=True)
    with open('result_updated.pred', 'a') as fileOut:
        for i in sorted([j+[i+1] for i,j in enumerate(tmp)],key=lambda x:(x[0],int(x[1].split('C')[-1]))):
            print("\t".join(map(str,i)),file=fileOut)
            #print("\t".join(map(str,i)))

print ("PRED file is updated")
end_exe=datetime.datetime.now()
total=end_exe-start_exe
print (total.total_seconds(),"seconds")