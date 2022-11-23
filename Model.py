from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from data_methods import getAllData, overSampling, nlp

if __name__ == '__main__':
    data = getAllData()
    # content = nlp(data['Text'])
    content = data['Text']
    label = data['Type']
    print("NLP processed complete")
    
    content, label = overSampling(content, label)
    cont_train, cont_test, lab_train, lab_test = train_test_split(content, label, test_size=0.3, random_state=233)
    
    cv = CountVectorizer()
    # Use CountVectorizer to obtain a vector of word counts.
    cont_train = cv.fit_transform(cont_train)
    cont_test = cv.transform(cont_test)
    print("Count Vectorizer Complete")
    
    # SVC model fitting and scoring
    svcModel = SVC(kernel="rbf", random_state = 12)
    svcModel.fit(cont_train, lab_train)
    print(f"SVM Score of the model is {svcModel.score(cont_test, lab_test)}")
    
    # Simple Logistic Regression fitting and scoring
    LRModel = LR(max_iter=10000)
    LRModel.fit(cont_train, lab_train)
    print(f"Logistic Regression score of the model is {LRModel.score(cont_test, lab_test)}")

    #Simple Random Forest 
    RFModel = RandomForestClassifier(max_depth=2, random_state=0)
    RFModel.fit(cont_train, lab_train)
    print(f"Random Forest of the model is {RFModel.score(cont_test, lab_test)}")

    #Gaussian Naive Bayes
    gaussNB = GaussianNB()
    gaussNB.fit(cont_train.toarray(), lab_train)
    print(f"Gaussian Naive Bayes score of the model is {gaussNB.score(cont_test.toarray(), lab_test)}")

    #Multinomial Naive Bayes
    multiNB = MultinomialNB()
    multiNB.fit(cont_train.toarray(), lab_train)
    print(f"Multinomial Naive Bayes score of the model is {multiNB.score(cont_test.toarray(), lab_test)}")                

    #Complement Naive Bayes
    compNB = ComplementNB()
    compNB.fit(cont_train.toarray(), lab_train)
    print(f"Complement Naive Bayes score of the model is {compNB.score(cont_test.toarray(), lab_test)}")               

    #Bernoulli Naive Bayes
    berNB = BernoulliNB()
    berNB.fit(cont_train.toarray(), lab_train)
    print(f"Bernoulli Naive Bayes score of the model is {berNB.score(cont_test.toarray(), lab_test)}") 