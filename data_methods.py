# Pytorch Data Loader Tutorials
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/data/main/tutorial.html
import numpy as np
# import torchdata.datapipes as dp
import pandas as pd
import numpy as np
from collections import defaultdict


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