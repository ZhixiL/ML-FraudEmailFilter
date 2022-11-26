import argparse
from fraud_classifier_class import FraudClassifier
from collections import defaultdict

def runAll(FC, rand):
    FC.SVC_Score(rand)
    FC.LR_Score(rand)
    FC.RandForest_Score(rand)
    FC.NaiveBayes_Score()

def runMult(FC, times):
    acc_scores, f_scores = defaultdict(list), defaultdict(list)
    for _ in range(times):
        ac, f = FC.SVC_Score()
        acc_scores['SVM'].append(ac)
        f_scores['SVM'].append(f)
        ac, f = FC.LR_Score()
        acc_scores['LR'].append(ac)
        f_scores['LR'].append(f)
        ac, f = FC.RandForest_Score()
        acc_scores['Rand'].append(ac)
        f_scores['Rand'].append(f)
        res = FC.NaiveBayes_Score()
        for i, pair in enumerate(res):
            ac, f = pair
            acc_scores[f'NB{i}'].append(ac)
            f_scores[f'NB{i}'].append(f)
    print(acc_scores)
    for key in acc_scores:
        print(f"{key} average accuracy is {sum(acc_scores[key])/len(acc_scores[key])}")
        print(f"{key} average f-score is {sum(f_scores[key])/len(f_scores[key])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--NLP-Option", required=False, default=1, type=int,
        help="0:No NLP at all (Not Recommended), 1:Simplified NLP(Default), 2:Full NLP (Time Consuming)"
    )
    parser.add_argument(
        "--testsize", required=False, default=0.3, type=float,
        help="0.3 by default."
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
        "--over_sampling", required=False, default=False, 
        help="True to turn on over_sampling (False by default)"
    )
    parser.add_argument(
        "--models", required=True, type=int, help="-1:all models, 0:SVM, 1:LR, 2:RandForest, 3:Naive Bayes"
    )
    args, _ = parser.parse_known_args()
    
    FC = FraudClassifier()
    
    FC.getAllData(args.NLP_Option)
    
    if args.over_sampling:
        FC.overSampling()
    FC.split_vectorize(testSize=args.testsize, random=args.randomSplit)
    
    if False: # Generate average results
        runMult(FC, times=5)
        exit()

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
                FC.NaiveBayes_Score()