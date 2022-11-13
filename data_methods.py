# Pytorch Data Loader Tutorials
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/data/main/tutorial.html
import numpy as np
# import torchdata.datapipes as dp
import pandas as pd
import numpy as np
from collections import defaultdict
import random

def filter_for_data(filename):
    return "sample_data" in filename and filename.endswith(".csv")

def row_processer(row):
    return {"label": np.array(row[0], np.int32), "data": np.array(row[1:], dtype=np.float64)}

# def build_datapipes(root_dir="./data/"):
#     datapipe = dp.iter.FileLister(root_dir)
#     datapipe = datapipe.filter(filter_fn=filter_for_data)
#     datapipe = datapipe.open_files(mode='rt')
#     datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
#     # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
#     datapipe = datapipe.shuffle()
#     datapipe = datapipe.map(row_processer)
#     return datapipe
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

def obtainExtraData(nCount, cCount, pCount):       #obtain x amount of datasets from the available dataset, pcount = phishing count, ccount = commercial spam count, ncount = normal email count
    pdbyType = pd.read_csv("./data/PhishingEmailData.csv", encoding = "ISO-8859-1")
    SpamHamDS = pd.read_csv("./data/spam_ham_dataset.csv", encoding = "ISO-8859-1")
    CEmailDS = pd.read_csv("./data/commercial email.csv", encoding = "ISO-8859-1")
    SubjectList, TextList, TypeList = [], [], []
    PhishingSubject, PhishingMessage = list(pdbyType["Email_Subject"]), list(pdbyType["Email_Content"])
    SHText, SHLabel= list(SpamHamDS["text"]), list(SpamHamDS["label"])
    CommercialSubject, CommercialText = [], []
    SHSubject, SHMessage = [], []
    
    dataDict = defaultdict() 

    for i in range(len(SHText)):            
        if SHLabel[i] == "ham":
            temp = SHText[i].split('\n')
            SHSubject.append(temp[0][8:])
            SHMessage.append(SHText[i][len(temp[0]):])

    pos = random.sample(range(len(SHSubject)), nCount)
    for i in pos:
        SubjectList.append(SHSubject[i])
        TextList.append(SHMessage[i])
        TypeList.append(0)
    for i in range(len(CEmailDS["message"])):
        if CEmailDS["type"][i] == "0":
            temp = str(CEmailDS["message"][i]).split("\n")
            PhishingSubject.append(temp[0][8:])
            PhishingMessage.append(str(CEmailDS["message"][i])[len(temp[0]):])
        elif CEmailDS["type"][i] == "1":
            temp = str(CEmailDS["message"][i]).split("\n")
            CommercialSubject.append(temp[0][8:])
            CommercialText.append(str(CEmailDS["message"][i])[len(temp[0]):])
    pos = random.sample(range(len(PhishingSubject)), pCount)
    for i in pos:
        SubjectList.append(PhishingSubject[i])
        TextList.append(PhishingMessage[i])
        TypeList.append(2)
    pos = random.sample(range(len(CommercialSubject)), cCount)
    for i in pos:
        SubjectList.append(CommercialSubject[i])
        TextList.append(CommercialText[i])
        TypeList.append(1)
    dataDict["Subject"] = SubjectList
    dataDict["Text"] = TextList
    dataDict["Type"] = TypeList
    return dataDict
