import xml.etree.ElementTree as ET
from vsm import Vsm
import datetime

start_exe=datetime.datetime.now()
tree = ET.parse('test_input.xml')
root = tree.getroot()

for thread in root.findall("Thread"):
    question = thread.find('RelQuestion')
    question_id = question.attrib['RELQ_ID']
    question_text = ""
    #if question.find('RelQBody').text is not None:
        #question_text = question.find('RelQBody').text
    if question.find("RelQSubject").text is not None:
        question_text += (' ' + question.find("RelQSubject").text)
    if question.attrib["RELQ_CATEGORY"] is not None:
        question_text += (' ' + question.attrib["RELQ_CATEGORY"])

    if not question_text:
        print('skipping', question_id)
        continue

    vsm = Vsm(question_text,question_id,thread.findall('RelComment'))
    sorted_results = vsm.evaluate()
    with open('result107.pred', 'a') as fileOut:
        sorted_ar = sorted(sorted_results, key=lambda x: (x[0], int(x[1].split('C')[-1])))
        for i in sorted_ar:
            print("\t".join(map(str, i)), file=fileOut)

print ("PRED file is updated")
end_exe=datetime.datetime.now()
total=end_exe-start_exe
print (total.total_seconds(),"seconds")
