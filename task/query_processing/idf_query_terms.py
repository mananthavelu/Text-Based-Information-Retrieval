#Calculating the idf for a query
def idf_query1(dfw):
    idf=[]
    import math
    for item in dfw['df']:
        if item !=0:
            value=math.log(10/item)
        else:
            value=0
        idf.append(value)
    dfw["idf"]=idf
    return dfw