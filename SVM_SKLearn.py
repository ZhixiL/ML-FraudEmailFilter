import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from data_methods import get_initdata

if __name__ == '__main__':
    # print(dm.build_datapipes())
    data = get_initdata()
    
    title = data['Subject']
    content = data['Text']
    label = data['Type']

    # include label in the content
    for i, t in enumerate(title):
        content[i] = str(t) + "\n" + str(content[i])

    cont_train, cont_test, lab_train, lab_test = train_test_split(content, label, test_size=0.2, random_state=1)
    
    cv = CountVectorizer()
    # transforming content to integer features
    cont_train = cv.fit_transform(cont_train)
    cont_test = cv.transform(cont_test)
    
    model = SVC(kernel = "rbf", random_state = 12)
    model.fit(cont_train, lab_train)
    
    print(model.score(cont_test, lab_test))
    