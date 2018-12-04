EquivalentAspects={}
with open('Test Data/Features.txt','r') as inputfile:
    for line in inputfile:
      aspects=line.strip().split(',')
      for aspect in aspects:
        EquivalentAspects[aspect]=aspects[0]

with open('Test Data/Balloon Data.Tagged','w') as outputfile:
    with open('Test Data/Test Data.Parsed','r') as inputfile:
        for line in inputfile:
            line = line.lower()
            if (line.startswith('[t]') or line.startswith('*')):
                outputfile.write(line)
            else:
                outputline=''
                segments=line.split('##')
                i=0
                for aspect in EquivalentAspects:
                    if aspect in segments[1].split() and EquivalentAspects[aspect] not in outputline:
                        if i>0:
                          outputline+=','
                        i+=1
                        outputline+=EquivalentAspects[aspect]+'[@]'
                outputline=outputline+'##'+segments[1]
                outputfile.write(outputline)