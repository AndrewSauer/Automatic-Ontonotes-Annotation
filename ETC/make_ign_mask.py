import json
import pickle
import tensorflow.compat.v1 as tf
TRAIN_PATH="../docred_data/train_annotated.json"
VAL_PATH="../docred_data/dev.json"
NUM_ENTITIES=50
NUM_LABELS=96
LABEL_STRINGS=['P159','P17','P131','P150','P27','P569','P19','P172','P571','P576','P607','P30','P276','P1376','P206','P495','P551','P264','P527','P463','P175','P577','P161','P403','P20','P69','P570','P108','P166','P6','P361','P36','P26','P25','P22','P40','P37','P1412','P800','P178','P400','P937','P102','P585','P740','P3373','P1001','P57','P58','P272','P155','P156','P194','P241','P127','P118','P39','P674','P179','P1441','P170','P449','P86','P488','P1344','P580','P582','P676','P54','P50','P840','P136','P205','P706','P162','P710','P35','P140','P1336','P364','P737','P279','P31','P137','P112','P123','P176','P749','P355','P1198','P171','P1056','P1366','P1365','P807','P190']
train_data=json.load(open(TRAIN_PATH))
val_data=json.load(open(VAL_PATH))
train_triples=[]
for i in range(len(train_data)):
    print(i)
    doc=train_data[i]
    for rel in doc['labels']:
        for hmention in doc['vertexSet'][rel['h']]:
            for tmention in doc['vertexSet'][rel['t']]:
                triple=[hmention['name'],tmention['name'],rel['r']]
                if not triple in train_triples:
                    train_triples.append(triple)
print("Train triples:")
print(len(train_triples))
ign_mask=tf.ones(tf.constant([len(val_data),NUM_ENTITIES,NUM_ENTITIES,NUM_LABELS]),"int32")
indices=[]
updates=[]
for i in range(len(val_data)):
    print(i)
    doc=val_data[i]
    for rel in doc['labels']:
        if not rel['r'] in LABEL_STRINGS:
            print("Fatal error! "+rel['r']+" not in LABEL_STRINGS!")
        for hmention in doc['vertexSet'][rel['h']]:
            for tmention in doc['vertexSet'][rel['t']]:
                triple=[hmention['name'],tmention['name'],rel['r']]
                if triple in train_triples:
                    indices.append([i,rel['h'],rel['t'],LABEL_STRINGS.index(rel['r'])])
                    print(triple)
                    updates.append(0)
                break;
            if [i,rel['h'],rel['t']] in indices:
                break;

ign_mask=tf.tensor_scatter_nd_update(ign_mask,tf.constant(indices),tf.constant(updates))
print("REPEATS:")
print(len(updates))
pickle.dump(ign_mask,open("ign_mask.pickle",'wb'))
#
