#Calculation of the tf-weight terms
def tf_weight(words_raw):    
    dict_tf_wt={}
    import math
    for key,val in words_raw.items():
        if val!=0:
            value=1+(math.log(val))
        else:
            value=0
        dict_tf_wt[key]=value
    return dict_tf_wt