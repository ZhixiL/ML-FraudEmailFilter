from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
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
    def SVC_Score(self):
        print('SVM Started')
        svcModel = SVC(kernel="rbf", random_state = 12)
        svcModel.fit(self.cont, self.lab)
        print(f"SVM Score of the model is {svcModel.score(self.cont_test, self.lab_test)}")
        
    # Simple Logistic Regression fitting and scoring
    def LR_Score(self):
        print("Logistic Regression Started")
        LRModel = LR(max_iter=10000)
        LRModel.fit(self.cont, self.lab)
        print(f"Logistic Regression score of the model is {LRModel.score(self.cont_test, self.lab_test)}")
    
    def RandForest_Score(self):
        print("Random Forest Started")
        RFModel = RandomForestClassifier(max_depth=2, random_state=0)
        RFModel.fit(self.cont, self.lab)
        print(f"Random Forest of the model is {RFModel.score(self.cont_test, self.lab_test)}")