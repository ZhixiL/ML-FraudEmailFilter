import argparse
from fraud_classifier_class import FraudClassifier

def runAll(FC):
    FC.SVC_Score()
    FC.LR_Score()
    FC.RandForest_Score()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--useNLP", required=False, default=False, help="Run complicate NLP instead of simple NLP"
    )
    parser.add_argument(
        "--data", required=False, default=0, help="0:All availiable data, 1:Base Only"
    )
    parser.add_argument(
        "--over_sampling", required=False, default=0, help="0:All availiable data, 1:Base Only"
    )
    parser.add_argument(
        "--models", required=True, type=int, help="-1:all models, 0:SVM, 1:LR, 2:RandForest, 3:Naive Bayes"
    )
    args, _ = parser.parse_known_args()
    
    FC = FraudClassifier()
    
    if args.data == 0:
        FC.getAllData(args.useNLP)
    else:
        print("get partial")
        
    FC.split_vectorize()
    if args.over_sampling:
        FC.overSampling()
        
    if args.models == -1:
        runAll(FC)
    else:
        match args.models:
            case 0:
                FC.SVC_Score()
            case 1:
                FC.LR_Score()
            case 2:
                FC.RandForest_Score()
            case 3:
                print('add NB here')