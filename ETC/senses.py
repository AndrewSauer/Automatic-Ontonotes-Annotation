import json,pickle
senses=[]
m=0
for i in range(3053):
    onf=json.load(open("../DocREDtrain/train_annotated"+str(i)+".final.json"))
    wsd=onf['wsd']
    if len(wsd)>m:
        m=len(wsd)
    for sense in wsd:
        sense=sense[2]
        if not sense in senses:
            senses.append(sense)
for i in range(998):
    onf=json.load(open("../DocREDdev/dev"+str(i)+".final.json"))
    wsd=onf['wsd']
    if len(wsd)>m:
        m=len(wsd)
    for sense in wsd:
        sense=sense[2]
        if not sense in senses:
            senses.append(sense)
"""for i in range(3053):
    onf=json.load(open("../REDocRED/train_revised"+str(i)+".final.json"))
    wsd=onf['wsd']
    if len(wsd)>m:
        m=len(wsd)
    for sense in wsd:
        sense=sense[2]
        if not sense in senses:
            senses.append(sense)
for i in range(500):
    onf=json.load(open("../REDocRED/dev_revised"+str(i)+".final.json"))
    wsd=onf['wsd']
    if len(wsd)>m:
        m=len(wsd)
    for sense in wsd:
        sense=sense[2]
        if not sense in senses:
            senses.append(sense)"""
print(len(senses))
print(m)
pickle.dump(senses,open("senses.pickle",'wb'))
