

with open('Test Data/Test Data.Parsed','w') as outputfile:
    with open('Test Data/Balloon.txt','r') as inputfile:
        for line in inputfile:
            if len(line) > 0 :
                for sentence in line.split('.'):
                    if len(sentence.strip())>0:
                        outputfile.write('##'+sentence.strip()+'\n')