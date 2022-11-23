from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from data_methods import getAllData, overSampling, nlp

class FraudClassifier:
    def __init__(self):
        self.data = []
        self.content = []
        self.label = []
    
    def getAllData(self, nlpOp = False):
        self.data = getAllData()
        self.content = self.data['Text']
        if nlpOp:
            self.content = nlp(self.content)
            print("NLP Complete")
        self.label = self.data['Type']
        print("Get All Data Complete")
    
    def overSampling(self):
        self.content, self.label = overSampling(self.content, self.label)
        print("Oversampling Complete")
    
    def split_vectorize(self, testSize = 0.3, random = None):
        self.cont, self.cont_test, self.lab, self.lab_test = train_test_split(
            self.content, self.label, test_size=testSize, random_state=random)
        # Use CountVectorizer to obtain a vector of word counts.
        cv = CountVectorizer()
        self.cont = cv.fit_transform(self.cont)
        self.cont_test = cv.transform(self.cont_test)
        print("Split and Count Vectorizer Complete")
        
    # SVC model fitting and scoring
    def SVC_Score(self, random = None):
        print('SVM Started')
        svcModel = SVC(kernel="rbf", random_state = random)
        svcModel.fit(self.cont, self.lab)
        y_pred = svcModel.predict(self.cont_test)
        score = f1_score(self.lab_test, y_pred, average='weighted')
        print(f"SVM Score of the model is {svcModel.score(self.cont_test, self.lab_test)}")
        print(f"SVM F1 score is {score}")
        
    # Simple Logistic Regression fitting and scoring
    def LR_Score(self, random = None):
        print("Logistic Regression Started")
        LRModel = LR(max_iter=10000, random_state = random)
        LRModel.fit(self.cont, self.lab)
        y_pred = LRModel.predict(self.cont_test)
        score = f1_score(self.lab_test, y_pred, average='weighted')
        print(f"Logistic Regression score of the model is {LRModel.score(self.cont_test, self.lab_test)}")
        print(f"Logistic Regression F1 score is {score}")
    
    # Random Forest Classifier
    def RandForest_Score(self, random = None):
        print("Random Forest Started")
        RFModel = RandomForestClassifier(random_state = random)
        RFModel.fit(self.cont, self.lab)
        y_pred = RFModel.predict(self.cont_test)
        score = f1_score(self.lab_test, y_pred, average='weighted')
        print(f"Random Forest of the model is {RFModel.score(self.cont_test, self.lab_test)}")
        print(f"Random Forest F1 score is {score}")
    
    def NaiveBayes_Score(self):
        print("Gaussian Naive Bayes Started")
        gaussNB = GaussianNB()
        gaussNB.fit(self.cont.toarray(), self.lab)
        y_pred = gaussNB.predict(self.cont_test.toarray())
        score = f1_score(self.lab_test, y_pred, average='weighted')
        print(f"Gaussian Naive Bayes score of the model is {gaussNB.score(self.cont_test.toarray(), self.lab_test)}")
        print(f"Gaussian NB F1 score is {score}")

        print("Multinomial Naive Bayes Started")
        multiNB = MultinomialNB()
        multiNB.fit(self.cont.toarray(), self.lab)
        y_pred = multiNB.predict(self.cont_test.toarray())
        score = f1_score(self.lab_test, y_pred, average = 'weighted')
        print(f"Multinomial Naive Bayes score of the model is {multiNB.score(self.cont_test.toarray(), self.lab_test)}")
        print(f"Multinomial NB F1 score is {score}")

        print("Complement Naive Bayes Started")
        compNB = ComplementNB()
        compNB.fit(self.cont.toarray(), self.lab)
        y_pred =  compNB.predict(self.cont_test.toarray())
        score = f1_score(self.lab_test, y_pred, average = 'weighted')
        print(f"Complement Naive Bayes score of the model is {compNB.score(self.cont_test.toarray(), self.lab_test)}")
        print(f"Complement NB F1 score is {score}")
        
        print("Bernoulli Naive Bayes Started")
        berNB = BernoulliNB()
        berNB.fit(self.cont.toarray(), self.lab)
        y_pred =  berNB.predict(self.cont_test.toarray())
        score = f1_score(self.lab_test, y_pred, average = 'weighted')
        print(f"Bernoulli Naive Bayes score of the model is {berNB.score(self.cont_test.toarray(), self.lab_test)}")
        print(f"Bernoulli NB F1 score is {score}")