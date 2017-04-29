import pandas as pd
import string

import processing as pr
import processing_word as prw

from sklearn.feature_extraction.text import CountVectorizer

class Vsm:

    def __init__(self,question, question_id,comments):
        self.translate_table = dict((ord(char), None) for char in string.punctuation)
        self.query = self.process_text(question)
        self.query_id = question_id
        self.documents = []
        self.documents_id = []
        for answer in comments:
            self.documents_id.append(answer.attrib['RELC_ID'])
            answer_text = answer.find('RelCText').text
            self.documents.append(self.process_text(answer_text))

    def evaluate(self):

        # Creating a wordset
        counter = prw.word_counter(self.documents)
        unique_words = prw.unique_words(counter)
        wordset = prw.wordset(self.query, unique_words)

        # Calculating tf-idf for question
        table_query = self.calculate_tfidf_query(wordset)

        # Normalization of tf-idf weights
        table_doc =  self.calculate_norm_doc()

        # Product
        columns = table_doc.columns[:]
        for item in columns:
            table_doc[item] *= table_query['nor_q']

        # Replacing the NaN values with Zero's
        doc_wt_corrected = table_doc.fillna('0')

        # Converting to Float datatype - This is to avoid the empty cosine score
        check = doc_wt_corrected.convert_objects(convert_numeric=True)

        #Summing up the dot products of Question and comments
        cos_score = list(check.sum(axis=0))

        #Combining the Comments ID's with the respective cosine score
        dictionary = dict(zip(self.documents_id, cos_score))

        scores = {}
        scores.update(dictionary)

        i = 1
        list_result = []
        for item in sorted(dictionary.keys()):
            list_inter = [self.query_id, item, i, dictionary[item], "true"]
            list_result.append(list_inter)
            i = i + 1

        # Returning the items of list with contents of output files before sorting 
        return sorted(list_result, key=lambda x: float(x[2]), reverse=True)

    def process_text(self, text):

        # Removing the punctuations
        text_without_punc = text.translate(self.translate_table)

        # Tokenization of the Question
        tokenized_text = pr.tokenization(text_without_punc)

        # Removing the stop words from the Question
        text_without_stop_w = pr.stop_words(tokenized_text)

        # Stemming the words in Question
        stemmed_text = pr.stemmer(text_without_stop_w)

        return stemmed_text

    def calculate_tfidf_query(self,wordset):

        # Calculating the raw terms in a question
        tf_raw = pr.tf_raw_query(self.query, wordset)
        
        #Converting the raw-terms into weights
        tf_wt = pr.tf_weight(tf_raw)

        # Calculating the document frequency for terms in terms index
        df_query = pr.calculate_document_frequency(list(wordset), self.query)

        # Creating a Dataframe with tf-raw, tf-wt, df as columns
        dframe = pd.DataFrame.from_dict([tf_raw, tf_wt, df_query], orient='columns')
        df = dframe.transpose()
        summary_of_results = df.rename(columns={0: 'tf_raw', 1: 'tf_weight', 2: 'df'})

        # Calculating the IDF for the query
        table_query = pr.calculate_idf_query(summary_of_results)
        table_query["wt"] = table_query.idf * table_query.tf_weight

        # Normalization
        table_query["nor_q"] = pr.norm_weight(table_query["wt"])
        return table_query

    def calculate_norm_doc(self):

        # Calculating the raw-terms for all the comments
        vect = CountVectorizer()
        text = pd.Series(self.documents).str.join(' ')
        X = vect.fit_transform(text)
        r = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
        doc_raw_terms = r.transpose()
        doc_weights = doc_raw_terms.applymap(pr.log_wt)

        # Adding the column names to the Dataframe
        doc_weights.columns = ["tf_doc_wt_1", "tf_doc_wt_2", "tf_doc_wt_3", "tf_doc_wt_4", "tf_doc_wt_5",
                                    "tf_doc_wt_6", "tf_doc_wt_7", "tf_doc_wt_8", "tf_doc_wt_9", "tf_doc_wt_10"]

        # Normalization of all the comments weights
        counter = 1
        for item in doc_weights:
            sum2 = 0.0
            norm_q_d = pr.norm_weight(doc_weights[item])
            s = str(counter)
            doc_weights[s] = norm_q_d
            counter += 1

        # Keeping the column weights and removing the remaining columns
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        doc_weights.drop(doc_weights.columns[cols], axis=1, inplace=True)
        return doc_weights







