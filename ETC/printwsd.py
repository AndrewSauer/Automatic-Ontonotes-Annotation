import sys,json
from nltk.corpus import wordnet as wn

filename=sys.argv[1]
obj=json.load(open(filename))
for sent in obj['sentences']:
    print(sent)
for i in range(len(obj['wsd'])):
    sense=obj['wsd'][i][2]
    synset=wn.synset_from_sense_key(sense)
    print(sense)
    print(synset.definition())
