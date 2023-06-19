import json,pickle
senses=[]
for i in range(3053):
    onf=json.load(open("../DocREDtrain/train_annotated"+str(i)+".final.json"))
    srl=onf['srl']
    for j in range(len(srl)):
        for sense in srl[j]['plemma_ids']:
            if not sense in senses:
                senses.append(sense)
for i in range(998):
    onf=json.load(open("../DocREDdev/dev"+str(i)+".final.json"))
    srl=onf['srl']
    for j in range(len(srl)):
        for sense in srl[j]['plemma_ids']:
            if not sense in senses:
                senses.append(sense)
print(len(senses))
pickle.dump(senses,open("frames.pickle",'wb'))
