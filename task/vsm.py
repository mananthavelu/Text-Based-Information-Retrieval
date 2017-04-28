import pandas as pd
import datetime
import string
from query_processing import tokenization
from query_processing import stop_words
from query_processing import stemmer
from general_processing import words_counter
from general_processing import unique_words_list
from general_processing import wordsets
from query_processing import tf_raw_query_terms
from query_processing import tf_weight_terms
from query_processing import document_frequency
from query_processing import idf_query_terms
from query_processing import normalization_query
import processing as pr


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
        # Creating a wordset for the comments
        counter = words_counter.word_counter(self.documents)
        unique_words = unique_words_list.unique_words(counter)
        wordset = wordsets.wordset(self.query, unique_words)

        norm_query = self.calculate_tfidf_norm_query(wordset)
        Comments_weights =  self.calculate_norm_doc(norm_query)

        # product
        columns = Comments_weights.columns[:]
        for item in columns:
            Comments_weights[item] *= norm_query['nor_q']
        # Replacing the NaN values with Zero's

        Comments_weights_corrected = Comments_weights.fillna('0')

        # Converting to Float datatype - This is to avoid the empty cosine score
        Check = Comments_weights_corrected.convert_objects(convert_numeric=True)

        Cosine_Score = Check.sum(axis=0)

        scoreee = list(Cosine_Score)

        dictionary = dict(zip(self.documents_id, scoreee))
        scores = {}
        scores.update(dictionary)

        i = 0
        list_result = []
        for item in sorted(dictionary.keys()):
            list_3 = [self.query_id, item, i, dictionary[item], "true"]
            list_result.append(list_3)
            i = i + 1
        return sorted(list_result, key=lambda x: float(x[2]), reverse=True)

    def process_text(self, text):
        text_without_punc = text.translate(self.translate_table)
        # Tokenization of the Question
        tokenized_text = tokenization.tokenization(text_without_punc)
        # Removing the stop words from the Question
        text_without_stop_w = stop_words.stop_words(tokenized_text)
        # Stemming the words in Question
        stemmed_text = stemmer.stemmer(text_without_stop_w)

        return stemmed_text

    def calculate_tfidf_norm_query(self,wordset):
        # Calculating the raw terms in the comments
        tf_raw = tf_raw_query_terms.tf_raw_query(self.query, wordset)
        tf_wt = tf_weight_terms.tf_weight(tf_raw)

        df_query = document_frequency.dfdfdf(list(wordset), self.query)

        # Appending the tf-raw, tf-wt, df into a dataframe
        dframe = pd.DataFrame.from_dict([tf_raw, tf_wt, df_query], orient='columns')
        df = dframe.transpose()
        summary_of_results = df.rename(columns={0: 'tf_raw', 1: 'tf_weight', 2: 'df'})

        # Calculating the IDF for the query
        idf_query = idf_query_terms.idf_query1(summary_of_results)
        idf_query["wt"] = idf_query.idf * idf_query.tf_weight

        # Normalization
        return normalization_query.norm_q(idf_query)

    def calculate_norm_doc(self,norm_query):
        # Calculating the raw-terms for all the comments
        vect = CountVectorizer()

        text = pd.Series(self.documents).str.join(' ')
        X = vect.fit_transform(text)

        r = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
        Comments_raw_terms = r.transpose()
        Comments_weights = Comments_raw_terms.applymap(pr.log_wt)

        # Adding the column names to the Dataframe
        Comments_weights.columns = ["tf_doc_wt_1", "tf_doc_wt_2", "tf_doc_wt_3", "tf_doc_wt_4", "tf_doc_wt_5",
                                    "tf_doc_wt_6", "tf_doc_wt_7", "tf_doc_wt_8", "tf_doc_wt_9", "tf_doc_wt_10"]

        # Normalization of all the comments weights

        counter = 1
        for item in Comments_weights:
            sum2 = 0.0
            for items in Comments_weights[item]:
                sq1 = items * items
                sum2 += sq1
                import math
                norm_q_d = []
                for itemw in Comments_weights[item]:
                    if itemw == 0:
                        deno = 0
                    else:
                        deno = itemw / math.sqrt((sum2))
                    norm_q_d.append(deno)
            s = str(counter)
            Comments_weights[s] = norm_q_d
            counter += 1

        # Keeping the column weights
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Comments_weights.drop(Comments_weights.columns[cols], axis=1, inplace=True)
        return Comments_weights







