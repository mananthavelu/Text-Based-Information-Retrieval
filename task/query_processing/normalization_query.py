#Normalizing the weights
def norm_q(df_norm):    
    sum1=0.0
    for item in df_norm['wt']:
        sq=item*item
        sum1+=sq
    import math
    norm_q=[]
    for itemm in df_norm['wt']:
        if itemm==0:
            deno=0
        else:
            deno=itemm/(math.sqrt((sum1)))
        norm_q.append(deno)
    df_norm["nor_q"]=norm_q
    return df_norm