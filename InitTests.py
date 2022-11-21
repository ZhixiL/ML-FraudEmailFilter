from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from data_methods import get_initdata, obtainExtraData

if __name__ == '__main__':
    data = get_initdata()
    extraData = obtainExtraData(200, 200, 200)
    for key in data:
        data[key] += extraData[key]
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
    
    # SVC model fitting and scoring
    svcModel = SVC(kernel="rbf", random_state = 12)
    svcModel.fit(cont_train, lab_train)
    print(f"SVM Score of the model is {svcModel.score(cont_test, lab_test)}")
    
    # Simple Logistic Regression fitting and scoring
    LRModel = LR(max_iter=10000)
    LRModel.fit(cont_train, lab_train)
    print(f"Logistic Regression score of the model is {LRModel.score(cont_test, lab_test)}")