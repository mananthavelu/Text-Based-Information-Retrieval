import xml.etree.ElementTree as ET
from vsm import Vsm
import datetime

start_exe=datetime.datetime.now()

# Importing the data
tree = ET.parse('test_input.xml')
root = tree.getroot()

# Iteration over every thread
for thread in root.findall("Thread"):

    # Accessing the question content and ID
    question = thread.find('RelQuestion')
    question_id = question.attrib['RELQ_ID']
    question_text = ""
    if question.find('RelQBody').text is not None:
        question_text = question.find('RelQBody').text
    if question.find("RelQSubject").text is not None:
        question_text += (' ' + question.find("RelQSubject").text)
    if question.attrib["RELQ_CATEGORY"] is not None:
        question_text += (' ' + question.attrib["RELQ_CATEGORY"])

    # Skipping the Question in case of empty question content
    if not question_text:
        print('skipping', question_id)
        continue

    # Implementation of Vector Space Model
    vsm = Vsm(question_text,question_id,thread.findall('RelComment'))

    # Sorting the results and assigning the score for each question and comment pair
    # Saving to an output text file

    sorted_results = vsm.evaluate()
    with open('unsupervised_rank_marimuthu_ananthavelu123.txt', 'a') as fileOut:
        sorted_ar = sorted(sorted_results, key=lambda x: (x[0], int(x[1].split('C')[-1])))
        for i in sorted_ar:
            print("\t".join(map(str, i)), file=fileOut)

#Printing execution time
print ("Output file is updated")
end_exe=datetime.datetime.now()
total=end_exe-start_exe
print ("Execution time is,",total.total_seconds(),"seconds")
