from fraud_classifier_class import FraudClassifier

if __name__ == '__main__':
    FC = FraudClassifier()
    FC.getAllData()
    FC.overSampling()
    FC.split_vectorize()
    FC.SVC_Score()
    FC.LR_Score()
    FC.RandForest_Score()