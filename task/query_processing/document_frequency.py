def dfdfdf(ww,dd):
    dict_dfdf={}
    for item in ww:
        #print (item)
        co=0
        for items in dd:
            #print (items)
            if item in items:
                co+=1
            else:
                pass
            dict_dfdf[item]=co
            #print ('First word')
    return dict_dfdf