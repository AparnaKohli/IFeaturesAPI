from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import glob
import os
import json

import math
import nltk
import nltk.data
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import numpy as np, numpy.random

from . import Stemming
from . import NDCG

# import Stemming
# import NDCG
from django.core.files.storage import FileSystemStorage


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Create your views here

@api_view(["POST"])

def ImplicitFeatures(request):
    try:
        files = glob.glob('inputfiles/*')
        for f in files:
            os.remove(f)
        if request.method == 'POST':
           uploaded_file1 = request.FILES['TextDataset']
           uploaded_file2 = request.FILES['SelectedFeatures']
           print("File1 Name:",uploaded_file1.name,"  ","File1 Size:",uploaded_file1.size)
           print("File2 Name",uploaded_file2.name,"  ","File2 Size",uploaded_file2.size)
           fs = FileSystemStorage()
           fs.save(uploaded_file1.name, uploaded_file1)
           fs.save(uploaded_file2.name, uploaded_file2)
        else:
           print("Wrong request method")
           
        # Identify different files and where you want to store the results
           
        Data_Directory='iPod/'
        DataFile='inputfiles/'+uploaded_file1.name
        Results_Directory='iPod/EM_Sentence/'
        EquivalenceFile= 'inputfiles/'+uploaded_file2.name
        
        EquivalentAspects={} # Create a list called Equivalent Aspects 
        with open(EquivalenceFile,'r') as inputfile:
            for line in inputfile:
              aspects=line.strip().split(',') # In selected features each line is split into different parts using a comma. Returns a list of strings which is stored in aspect. 
              for aspect in aspects: # Use this as an iterator to go over every string in the list of strings returned above 
                EquivalentAspects[aspect]=aspects[0] # the first item in the list is assigned to the appropriate index in the list
                # print(aspects[0])
        # print ("EquivalentAspects",EquivalentAspects)
        
        with open(Data_Directory+'iPod.data','w') as outputfile: # Output file created and output inserted in here
            with open(DataFile,'r') as inputfile: # Raw data file with all the reviews (but for now iPod.final already has the format we are trying to insert into output file)
                for line in inputfile:
                    if (line.startswith('[t]') or line.startswith('*')):
                        outputfile.write(line)
                    else:
                        outputline=''
                        segments=line.split('##')
                        #print("Segments:",segments)
                        i=0
                        for aspect in EquivalentAspects:
                            if aspect in segments[1].split() and EquivalentAspects[aspect] not in outputline:
                                if i>0:
                                  outputline+=','
                                i+=1
                                outputline+=EquivalentAspects[aspect]+'[@]'
                        outputline=outputline+'##'+segments[1]
                        outputfile.write(outputline)
                        #print(outputline)
        
        punctuation = "'.,:;!?1234567890"
        StopWords=[]
        
        class aspectSentiment:
          def __init__(self, aspect, sentiment, implicit):
            self.aspect = aspect # Self is an instance of a class in Python. Init is the constructor
            self.sentiment = sentiment
            self.implicit=implicit

        
        def remove_punctuation(input_string):
          input_string1=''
          for word in input_string.split():# Split the string to get a list of strings
            if word not in StopWords:
              input_string1 = input_string1+' '+word # if this is not a stopword, keep adding the words separated by a space.
          for item in punctuation: # if the string is in punctuation, we replace by space
            input_string1 = input_string1.replace(item, ' ')
          return input_string1

        Precision=[]
        Recall=[]
        F_Measure=[]
        GlobalNDCG=[]


        for GlobalIteration in range(0,1):
          AllData=[]
          TrainingData=[]
          TestData=[]
          Reviews=[]
          RawReviews=[]
          GroundTruth=[]
          Collection={}
          TopicModel={}
          ExplicitFeatures={}
          IDF={}
          WordcountPerLine={}
          BackgroundProbability={}
          EquivalentAspects={}
        
          TotalNoOfReviews=0
          TotalNoOfWords=0
        
          with open(EquivalenceFile,'r') as inputfile:
            for line in inputfile:
              aspects=line.strip().split(',')
              for aspect in aspects:
                EquivalentAspects[aspect]=aspects[0] # The Equivalence File will gave a list of all the features which need to be included for the product

          
          with open('iPod/lemur.stopwords','r') as inputfile:
            for line in inputfile:
              StopWords.append(line.strip()) # Add the stopwords to the StopWords list
          
          with open(DataFile,'r') as inputfile:
            for line in inputfile:
              if (line.startswith('[t]') or line.startswith('*')):
                TotalNoOfReviews+=1 # Count number of reviews 
              AllData.append(line) # Append data in the AllData list
        
          for line in AllData:
            TrainingData.append(Stemming.perform_Stemming(line)) # Do Stemming for every line in the main database

          NoOfReviews=-1
          i=0
          j=0
          for line in TrainingData:
            i+=1
            TotalNoOfWords+=len(line.split()) #no of words in each line in training data
            if (line.startswith('[t]') or line.startswith('*')):
              NoOfReviews+=1
              Reviews.append(list()) # Add an empty list to the reviews , raw reviews, ground truth for each iteration
              RawReviews.append(list())
              GroundTruth.append(list())
              GroundTruth[NoOfReviews].append(list())
              #print("before for loop:",Reviews[0])
              j=-1
            else:      
              
              GroundTruth[NoOfReviews].append(list())
              review_with_aspect=line.replace('(','').replace(')','').replace('-','').split('##')
        
              review_Sentence=remove_punctuation(review_with_aspect[1])
              #print("Review Sentence",review_Sentence)
              if review_Sentence.strip()=='':
                continue
              j+=1
              RawReviews[NoOfReviews].append(line) #put one review of this iteration into rawreviews - each line of a review as an item in the list
              Reviews[NoOfReviews].append({}) 
              #print("Raw Reviews",RawReviews[NoOfReviews])
              #print("Reviews",Reviews[NoOfReviews])
              for word in review_Sentence.split(): #review_sentence has undergone transformation before for replacing symbols, removing punctuations and string spaces before and after sentence
                if word in Collection: #to get count of all words in all the reviews into collection - acts as collection of the data set 
                  Collection[word]+=1
                else:
                  Collection[word]=1
                if word in Reviews[NoOfReviews][j]: #Reviews is a list of all reviews with each review as a dictionary and each dictionary has words and count of each words from that review
                  Reviews[NoOfReviews][j][word]+=1
                  #print('1',Reviews[NoOfReviews][j][word])
                else:
                  Reviews[NoOfReviews][j][word]=1
                  #print('2',Reviews[NoOfReviews][j][word])
              #print("IDF",IDF)
              #print("Collection",Collection)
              for word in set(review_Sentence.split()): # count of all the words in the whole collection of reviews
                if word in IDF:
                  IDF[word]+=1
                else:
                  IDF[word]=1
              #print("len of review with aspect",len(review_with_aspect[0]))
              #print(review_with_aspect)
              #print("review with aspect[0]",review_with_aspect[0])
              if len(review_with_aspect[0])>0: #review with aspect[0] will have the explicit feature mentioned or null if no feature in that line. Can have multiple features delimited with comma.
                #print(review_with_aspect)
                #print("len of review with aspect",len(review_with_aspect[0]))
                #print("review with aspect[0]",review_with_aspect[0])
                explicitFeatures=review_with_aspect[0].split(',')
                for feature in explicitFeatures: #loop for each feature
                  feature_sentiment_context=feature.split('[')
                  sentiment_context=feature_sentiment_context[1].split('@')
                  if '[u]' not in feature:
                    if EquivalentAspects[feature_sentiment_context[0].strip()] in ExplicitFeatures: #count of explicit features in collection of all reviews
                      ExplicitFeatures[EquivalentAspects[feature_sentiment_context[0].strip()]]+=1
                    else:
                      ExplicitFeatures[EquivalentAspects[feature_sentiment_context[0].strip()]]=1
                    
                    if EquivalentAspects[feature_sentiment_context[0].strip()] not in TopicModel: #create a dictionary TopicModel for each explicit feature for lines with no explicit features mentioned 
                      TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]]={}
        
                    for word in review_Sentence.split(): #count of all the words (except stop words) in sentences that have explicit features tagged into the dictionary
                      if word in TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]]:
                        TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]][word]+=1
                      else:
                        TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]][word]=1

                    GroundTruth[NoOfReviews][j].append(aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()],sentiment_context[0],False))
                  else:

                    GroundTruth[NoOfReviews][j].append(aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()],sentiment_context[0],True)) #lines with explicit feature tags along with [u] having "True"


        
          TotalLines=i
        
          TotalAspectOccurance=0
          for aspect in ExplicitFeatures: #Total consolidated sum of all explicit features occurences
            #print("aspect:",aspect)
            TotalAspectOccurance+=ExplicitFeatures[aspect]
            #print("TotalAspectOccurance:",TotalAspectOccurance)
        
          for aspect in ExplicitFeatures: #distribution of feature occurences
            ExplicitFeatures[aspect]/=TotalAspectOccurance
            #print(ExplicitFeatures[aspect])

            
          for word in Collection: #distribution of all the words in collection
              BackgroundProbability[word]=Collection[word]/TotalNoOfWords
              #print(word,BackgroundProbability[word])
              
          for word in Collection: #words in collection that are not in TopicModel is 0
            for aspect in TopicModel:
              if word not in TopicModel[aspect]:
                TopicModel[aspect][word]=0
        
        
          for aspect in TopicModel: 
            for word in TopicModel[aspect]:
              TopicModel[aspect][word]=math.log(1+TopicModel[aspect][word])*math.log(1+(TotalLines/IDF[word]))
              #print(aspect,"",TopicModel[aspect][word])
        
        
        
          for aspect in TopicModel:
            sumOfTFIDF=sum(TopicModel[aspect].values())
            for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True):
              TopicModel[aspect][word]=(TopicModel[aspect][word]+1)/(sumOfTFIDF+len(Collection))
              
          #Topic Model File
          with open(Results_Directory+'TopicModel.txt','w') as outputFile:
            for aspect in TopicModel:
              for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True)[:10]:
                outputFile.write(aspect+', '+word+', '+str(TopicModel[aspect][word])+'\n')
              outputFile.write('\n\n\n')
              for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True)[:3]:
                SynonymSet=[]
                syns = wn.synsets(word)
                for s in syns:
                   for l in s.lemma_names():
                      SynonymSet.append(l)
                for s in SynonymSet:
                  if s not in TopicModel[aspect]:
                    TopicModel[aspect][s]=TopicModel[aspect][word]


          '''
          for word in sorted(BackgroundProbability, key=BackgroundProbability.get, reverse=True)[:10]:
              print(word, BackgroundProbability[word])
          '''
          #Explicit Features File
          with open(Results_Directory+'ExplicitFeatures.txt','w') as outputfile:
            for aspect in sorted(ExplicitFeatures, key=ExplicitFeatures.get, reverse=True):
                outputfile.write(aspect + ' : ' + str(ExplicitFeatures[aspect])+'\n')
        
        
          HP=[]
          HPB=[]
          PI=[]
          for reviewNum in range(0,len(Reviews)):
            HP.append(list())
            HPB.append(list())
            PI.append(list())
            for lineNum in range(0,len(Reviews[reviewNum])):
              HP[reviewNum].append({})
              HPB[reviewNum].append({})
              PI[reviewNum].append({})
              for word in Reviews[reviewNum][lineNum]:
                HP[reviewNum][lineNum][word]={}
                for aspect in TopicModel:
                  HP[reviewNum][lineNum][word][aspect]=0.0
                HPB[reviewNum][lineNum][word]=0.0
              RandomProbabilities=[np.random.dirichlet(np.ones(len(TopicModel)),size=1)[0]]
              myIndex=0
              for aspect in TopicModel:
                PI[reviewNum][lineNum][aspect]=RandomProbabilities[0][myIndex]
                myIndex+=1
              
        
        
        
          max_iter=50
          lambdaB=0.5
          dist_threshold=1e-6
        
          print("Starting EM-1------------------------")
        
          for i in range(max_iter):
               print('iteration: ' +str(i))
               i+=1
               #E-step
               print('E-step')
               for reviewNum in range(0,len(Reviews)):
                 for lineNum in range(0,len(Reviews[reviewNum])):
                    for word in Reviews[reviewNum][lineNum]:
                      mysum=0
                      #P(Z(d,w) = j) - Topic Model
                      for aspect in TopicModel:
                          mysum+=PI[reviewNum][lineNum][aspect]*TopicModel[aspect][word]
                      for aspect in TopicModel:
                          HP[reviewNum][lineNum][word][aspect]=PI[reviewNum][lineNum][aspect]*TopicModel[aspect][word]/mysum
                      
                      #P(Z(d,w) = B) - Background Model
                      HPB[reviewNum][lineNum][word]=(lambdaB*BackgroundProbability[word])/(lambdaB*BackgroundProbability[word]+((1-lambdaB)*mysum))
        
        
               print('M-step')
        
               #M-step
               #print('Computing PI')
               previousPI=[]
               for reviewNum in range(0,len(Reviews)):
                 previousPI.append(list())
                 for lineNum in range(0,len(Reviews[reviewNum])):
                   previousPI[reviewNum].append({})
                   for aspect in TopicModel:
                     previousPI[reviewNum][lineNum][aspect]=PI[reviewNum][lineNum][aspect]
        
               #PI(n+1) for d,j in M-Step
               for reviewNum in range(0,len(Reviews)):
                 for lineNum in range(0,len(Reviews[reviewNum])):
                    denom=0
                    for aspect in TopicModel:
                        for word in Reviews[reviewNum][lineNum]:
                            denom+=Reviews[reviewNum][lineNum][word]*(1-HPB[reviewNum][lineNum][word])*HP[reviewNum][lineNum][word][aspect]
                            
                    for aspect in TopicModel:
                        nom=0
                        for word in Reviews[reviewNum][lineNum]:
                            nom+=Reviews[reviewNum][lineNum][word]*(1-HPB[reviewNum][lineNum][word])*HP[reviewNum][lineNum][word][aspect]
                        try:
                          PI[reviewNum][lineNum][aspect]=nom/denom
                        except:
                          print(reviewNum,lineNum,aspect,nom,denom)
        
        
               dist=0.0
               for reviewNum in range(0,len(Reviews)):
                 for lineNum in range(0,len(Reviews[reviewNum])):
                    for aspect in TopicModel:
                         dist=dist+math.pow(PI[reviewNum][lineNum][aspect]-previousPI[reviewNum][lineNum][aspect],2)
               print('dist='+str(dist))
               if dist < dist_threshold:
                    break
                    
        
          #Calculating Precision, Recall, F-Measure, NDCG
          Precision.append([])
          Recall.append([])
          F_Measure.append([])
          GlobalNDCG.append([])
          ##Test Data Evaluation
          TPDict={}
          FNDict={}
          FPDict={}
          print("line 410")
        
          #for MyK in range(0,len(ExplicitFeatures)+1):
          MyK=0
          colorprobabilityOutput=0
          precisionOutput=0
          recallOutput=0
          fmeasureOutput=0
          ndcgOutput=0
          for ColorProbability in np.arange(0.35, 0.39, 0.05):
            Theta=0.0
            TP=0
            FP=0
            FN=0
            FrequentThreshold=0.2
            ndcgList=[]
        
            for reviewNum in range(0,len(Reviews)):
              for lineNum in range(0,len(Reviews[reviewNum])):
                if '[u]' in RawReviews[reviewNum][lineNum]:
                  ActualImplicitFeatureSet=set([])
                  ActualExplicitFeatureSet=set([])
                  for item in GroundTruth[reviewNum][lineNum]:
                    if item.implicit==True:
                      ActualImplicitFeatureSet.add(item.aspect)
                    else:
                      ActualExplicitFeatureSet.add(item.aspect)
                  #print('\n\n\n\n')
                  #print(RawReviews[reviewNum][lineNum])
                  #print('\n')
                  InferredImplicitFeatureSet=set([])
        
        
        
                  for aspect in sorted(PI[reviewNum][lineNum], key=PI[reviewNum][lineNum].get, reverse=True):
                    if PI[reviewNum][lineNum][aspect]>=ColorProbability and aspect not in ActualExplicitFeatureSet and aspect not in Reviews[reviewNum][lineNum] and ExplicitFeatures[aspect]>=FrequentThreshold and aspect not in InferredImplicitFeatureSet:
                        #print(word, aspect, HP[reviewNum][lineNum][word][aspect],HPB[reviewNum][lineNum][word])
                        InferredImplicitFeatureSet.add(aspect)
                          
                  tagged_list = []


                  # for aspect in sorted(PI[reviewNum][lineNum], key=PI[reviewNum][lineNum].get, reverse=True):
                  #   if aspect not in ActualExplicitFeatureSet and aspect not in Reviews[reviewNum][lineNum] and ExplicitFeatures[aspect]>=FrequentThreshold:
                  #     print(aspect,PI[reviewNum][lineNum][aspect])
                  #     print(Reviews[reviewNum][lineNum])
                  
                  for aspect in ExplicitFeatures:
                    if ExplicitFeatures[aspect]<FrequentThreshold or aspect in Reviews[reviewNum][lineNum]:
                      try:
                        ActualImplicitFeatureSet.remove(aspect)
                      except:
                        emni=1
                  #print('\nActual:\n--------------')
                  #for aspect in ActualImplicitFeatureSet:
                    #print(aspect)
        
                  if len(ActualImplicitFeatureSet)==0:
                    continue
                  
                  TP+=len(InferredImplicitFeatureSet.intersection(ActualImplicitFeatureSet))
                  FP+=len(InferredImplicitFeatureSet - ActualImplicitFeatureSet)
                  FN+=len(ActualImplicitFeatureSet - InferredImplicitFeatureSet)
        
                  for aspect in InferredImplicitFeatureSet.intersection(ActualImplicitFeatureSet):
                    if aspect not in TPDict:
                      TPDict[aspect]=1
                    else:
                      TPDict[aspect]+=1
        
                  for aspect in InferredImplicitFeatureSet - ActualImplicitFeatureSet:
                    if aspect not in FPDict:
                      FPDict[aspect]=1
                    else:
                      FPDict[aspect]+=1
        
                  for aspect in ActualImplicitFeatureSet - InferredImplicitFeatureSet:
                    if aspect not in FNDict:
                      FNDict[aspect]=1
                    else:
                      FNDict[aspect]+=1
                  
                  IdealFeatureList={}
                  InferredFeatureList={}
                  for aspect in PI[reviewNum][lineNum]:
                    if aspect in ActualImplicitFeatureSet:
                      IdealFeatureList[aspect]=1
                      InferredFeatureList[aspect]=PI[reviewNum][lineNum][aspect]
                    elif aspect not in ActualExplicitFeatureSet:
                      InferredFeatureList[aspect]=PI[reviewNum][lineNum][aspect]
                      IdealFeatureList[aspect]=0
                  currentNDCG=NDCG.compute_NDCG(IdealFeatureList,InferredFeatureList)
                  if currentNDCG!=-1:
                    ndcgList.append(currentNDCG)
            try:  
              Precision[GlobalIteration].append(TP/(TP+FP))
            except:
              Precision[GlobalIteration].append(0)
            try:
              Recall[GlobalIteration].append(TP/(TP+FN))
            except:
              Recall[GlobalIteration].append(0)
            try:
              F_Measure[GlobalIteration].append((2*Precision[GlobalIteration][MyK]*Recall[GlobalIteration][MyK])/(Precision[GlobalIteration][MyK]+Recall[GlobalIteration][MyK]))
            except:
              F_Measure[GlobalIteration].append(0)

            results_to_send = []
            aspect_dict = dict()

            for review in range(len(RawReviews)):
                for line in range(len(RawReviews[review])):
                    result = [RawReviews[review][line]]
                    for aspect in PI[review][line]:
                        if PI[review][line][aspect] >= FrequentThreshold:
                            result.append(aspect)
                            if aspect not in aspect_dict:
                                aspect_dict[aspect] = 1
                            else:
                                aspect_dict[aspect] += 1
                    results_to_send.append(result)

            aspect_rank = []
            for aspect in aspect_dict:
                aspect_rank.append((aspect_dict[aspect], aspect))

            aspect_rank.sort(reverse=True)


            print('P=='+str(ColorProbability))
            print("Precision:",Precision[GlobalIteration][MyK],"\nRecall:",Recall[GlobalIteration][MyK],"\nF-Measure:",F_Measure[GlobalIteration][MyK])
            print('NDCG:'+str(sum(ndcgList)/len(ndcgList)))
            print('###########################\n')
            GlobalNDCG[GlobalIteration].append(sum(ndcgList)/len(ndcgList))
            
            colorprobabilityOutput=str(ColorProbability)
            precisionOutput=str(Precision[GlobalIteration][MyK])
            recallOutput=str(Recall[GlobalIteration][MyK])
            fmeasureOutput=str(F_Measure[GlobalIteration][MyK])
            ndcgOutput=str(sum(ndcgList)/len(ndcgList))
            
            MyK+=1


        #Average precision, recall, ndcg for collection
        with open(Results_Directory+'ExplicitFeatures.txt','w') as Summaryfile:
          Summaryfile.write('P,Precision,Recall,F_measure,NDCG\n')
          col_avg_Precision = [ sum(x)/len(Precision) for x in zip(*Precision)]
          col_avg_Recall = [ sum(x)/len(Recall) for x in zip(*Recall)]
          col_avg_F_Measure = [ sum(x)/len(F_Measure) for x in zip(*F_Measure)]
          col_avg_GlobalNDCG = [ sum(x)/len(GlobalNDCG) for x in zip(*GlobalNDCG)]
          for i in range(0,len(col_avg_Precision)):
            Summaryfile.write(str(i*0.05)+','+str(col_avg_Precision[i])+','+str(col_avg_Recall[i])+','+str(col_avg_F_Measure[i])+','+str(col_avg_GlobalNDCG[i])+'\n')
            #print((str(i*0.05)+','+str(col_avg_Precision[i])+','+str(col_avg_Recall[i])+','+str(col_avg_F_Measure[i])+','+str(col_avg_GlobalNDCG[i])+'\n'))

        print(aspect_rank)
        print(results_to_send)

        responseData = {
                'P':colorprobabilityOutput,
                'Precision':precisionOutput,
                'Recall': recallOutput,
                'F-Measure': fmeasureOutput,
                'Aspect Counts': json.dumps(aspect_rank),
                'Tagged Reviews': json.dumps(results_to_send)
                }
        
        #return JsonResponse("P=" + colorprobabilityOutput + "Precision:" + precisionOutput+ "Recall:"+ recallOutput+ "F-Measure:"+ fmeasureOutput, safe=False)
        return JsonResponse(responseData,safe=False)

        
    except Exception as e:
         return JsonResponse("<p>Error: %s</p>" % str(e), safe = False )