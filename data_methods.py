# Pytorch Data Loader Tutorials
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/data/main/tutorial.html
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import string

def filter_for_data(filename):
    return "sample_data" in filename and filename.endswith(".csv")

def row_processer(row):
    return {"label": np.array(row[0], np.int32), "data": np.array(row[1:], dtype=np.float64)}

def get_initdata(): #return title, text and label
    pdbyType = pd.read_csv("./data/phishing_data_by_type.csv", encoding = "ISO-8859-1")
    dataDict = defaultdict()
    dataDict["Subject"] = list(pdbyType["Subject"])
    dataDict["Text"] = list(pdbyType["Text"])
    Type = []
    for i in pdbyType["Type"]:
        if i == "Fraud" or i == "Phishing":
            Type.append(2)
        elif i == "Commercial Spam":
            Type.append(1)
        else:
            Type.append(0)
    dataDict["Type"] = Type
    return dataDict

#obtain x amount of datasets from the available dataset, pcount = phishing count, ccount = commercial spam count, ncount = normal email count
def obtainExtraData(nCount, cCount, pCount):
    pdbyType = pd.read_csv("./data/PhishingEmailData.csv", encoding = "ISO-8859-1")
    SpamHamDS = pd.read_csv("./data/spam_ham_dataset.csv", encoding = "ISO-8859-1")
    CEmailDS = pd.read_csv("./data/commercial email.csv", encoding = "ISO-8859-1")
    SubjectList, TextList, TypeList = [], [], []
    PhishingSubject, PhishingMessage = list(pdbyType["Email_Subject"]), list(pdbyType["Email_Content"])
    SHText, SHLabel= list(SpamHamDS["text"]), list(SpamHamDS["label"])
    CommercialSubject, CommercialText = [], []
    SHSubject, SHMessage = [], []

    # PREPROCESS DATA
    # Split normal email into subject and text, while appending their type.
    dataDict = defaultdict()
    for i in range(len(SHText)):
        if SHLabel[i] == "ham":
            temp = SHText[i].split('\n')
            SHSubject.append(temp[0][8:])
            SHMessage.append(SHText[i][len(temp[0]):])
        
    # Select pCount phishing emails (most len(PhishingSubject)), cCount commercial emails (most len(CommercialSubject))
    for i in range(len(CEmailDS["message"])):
        if CEmailDS["type"][i] == "0":
            temp = str(CEmailDS["message"][i]).split("\n")
            PhishingSubject.append(temp[0][8:])
            PhishingMessage.append(str(CEmailDS["message"][i])[len(temp[0]):])
        elif CEmailDS["type"][i] == "1":
            temp = str(CEmailDS["message"][i]).split("\n")
            CommercialSubject.append(temp[0][8:])
            CommercialText.append(str(CEmailDS["message"][i])[len(temp[0]):])


    # SELECT DATA
    nCount, pCount, cCount = min(nCount, len(SHSubject)), min(pCount, len(PhishingSubject)), min(cCount, len(CommercialSubject))
    print(f"We have at most {len(SHSubject)} type 0, {len(CommercialSubject)} type 1, {len(PhishingSubject)} type 2.")
    
    # Select nCount dataset (max len(SHSubject))
    pos = random.sample(range(len(SHSubject)), nCount)
    for i in pos:
        SubjectList.append(SHSubject[i])
        TextList.append(SHMessage[i])
        TypeList.append(0)

    # Randomize then add them to the list to the return lists
    pos = random.sample(range(len(PhishingSubject)), pCount)
    for i in pos:
        SubjectList.append(PhishingSubject[i])
        TextList.append(PhishingMessage[i])
        TypeList.append(2)
    # Randomize commercial email to add to the list to return list
    pos = random.sample(range(len(CommercialSubject)), cCount)
    for i in pos:
        SubjectList.append(CommercialSubject[i])
        TextList.append(CommercialText[i])
        TypeList.append(1)
    dataDict["Subject"] = SubjectList
    dataDict["Text"] = TextList
    dataDict["Type"] = TypeList
    return dataDict

def MessageProcessing(message):
    for i in range(len(message)):
        message[i] = message[i].lower().replace('\n', ' ').replace(':', ' ').replace(';', ' ').replace('"', ' ').replace(",", ' ').replace(".", ' ').replace("?",' ').replace("!", ' ').replace("-", ' ')
        message[i] = "".join(x for x in message[i] if x not in string.punctuation).lower()
        message[i] = " ".join(message[i].split())
    return message

def getAllData():
    dataDict = defaultdict()
    Subject, Text, Type = [], [], []
    tempDict = get_initdata()       #get the initial list from the phishing_data_by_type dataset
    Subject.extend(tempDict["Subject"])     #extend the data returned from get_initdata into its respective categories
    Text.extend(tempDict["Text"])
    Type.extend(tempDict["Type"])
    tempDict2 = obtainExtraData(3672, 194, 693)      #3672 normal, 194 commercial, 693 phishing
    Subject.extend(tempDict2["Subject"])
    Text.extend(tempDict2["Text"])
    Type.extend(tempDict2["Type"])
    Message = []
    for i in range(len(Subject)):                   #some Subject were incorrect, this removes them
        if type(Subject[i]) == float:
            Subject[i] = ""
    Subject = MessageProcessing(Subject)            #process the messages to make them lowercase and remove unnecessary things
    Text = MessageProcessing(Text)
    for i in range(len(Subject)):
        Message.append(Subject[i] + " " + Text[i])          #combine Subject and Text into one list
    temp = list(zip(Message, Type))                         #zip the two to shuffle later
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    MessageRes, TypeRes = list(res1), list(res2)
    dataDict["Text"] = MessageRes
    dataDict["Type"] = TypeRes
    return dataDict

def overSampling(content, label):
    newCont, newLab = [], []
    finalItrs = []
    maxDp = max(Counter(label).values())
    dp = defaultdict(list)
    for i, lab in enumerate(label):
        dp[lab].append(i)
    for key in dp.keys():
        # this is the largest dataset case, we keep all the original data.
        if len(dp[key]) == maxDp:
            finalItrs += dp[key]
            continue
        # now we handle smaller dataset.
        cur = []
        while len(cur) < maxDp:
            cur.append(dp[key][random.randrange(0, len(dp[key]))])
        finalItrs += cur
    # randomize itr list so we don't have distinct section of different labels.
    random.shuffle(finalItrs)
    for i in finalItrs:
        newCont.append(content[i])
        newLab.append(label[i])
    return (newCont, newLab)