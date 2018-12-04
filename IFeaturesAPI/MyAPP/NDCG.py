import math


def compute_NDCG(ActualScore, predictedScore):
    IDCG=0
    i=1
    for item in sorted(ActualScore, key=ActualScore.get, reverse=True)[:5]:
        IDCG+=(math.pow(2,ActualScore[item])-1)/math.log2(i+1)
        i+=1

    DCG=0
    i=1
    for item in sorted(predictedScore, key=predictedScore.get, reverse=True)[:5]:
        DCG+=(math.pow(2,ActualScore[item])-1)/math.log2(i+1)
        i+=1
    if IDCG==0:
        #print('########################################################')
        #print(ActualScore)
        #print(predictedScore)
        return -1
    return (DCG/IDCG)
