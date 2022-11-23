import argparse
from fraud_classifier_class import FraudClassifier

def runAll(FC, rand):
    FC.SVC_Score(rand)
    FC.LR_Score(rand)
    FC.RandForest_Score(rand)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--useNLP", required=False, default=False,
        help="False by default, run complicate NLP instead of simple NLP (Extremely time consuming)"
    )
    parser.add_argument(
        "--testsize", required=False, default=0.3, help="0.3 by default."
    )
    parser.add_argument(
        "--randomSplit", required=False, type=int, default=None,
        help="Disabled by default, take in seed for random split between train and test."
    )
    parser.add_argument(
        "--randomState", required=False, type=int, default=0,
        help="Disabled by default, take in seed for model's random state."
    )
    parser.add_argument(
        "--over_sampling", required=False, default=0, help="0:All availiable data, 1:Base Only"
    )
    parser.add_argument(
        "--models", required=True, type=int, help="-1:all models, 0:SVM, 1:LR, 2:RandForest, 3:Naive Bayes"
    )
    args, _ = parser.parse_known_args()
    
    FC = FraudClassifier()
    
    FC.getAllData(args.useNLP)
    
    if args.over_sampling:
        FC.overSampling()
    FC.split_vectorize(testSize=args.testsize, random=args.randomSplit)

    if args.models == -1:
        runAll(FC, args.randomState)
    else:
        match args.models:
            case 0:
                FC.SVC_Score(args.randomState)
            case 1:
                FC.LR_Score(args.randomState)
            case 2:
                FC.RandForest_Score(args.randomState)
            case 3:
                print('add NB here')