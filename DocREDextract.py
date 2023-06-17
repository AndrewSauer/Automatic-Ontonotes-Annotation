#extract the training or dev data from REDocRED into txt files so they can be annotated
import json

with open("docred_data/dev.json",'r') as f:
    data=json.load(f);

fnum=0;
for doc in data:
    fpath="DocREDdev/dev"+str(fnum)+".txt";
    text="";
    for sent in doc['sents']:
        for word in sent:
            text+=word+' ';
        text+='\n\n';
    with open(fpath,'w') as f:
        print(text,file=f);
    text="";
    fnum+=1;
#Data is structured as a list of documents
#Each document has a VertexSet, a list of entities, each with a 'type', position in sentence 'pos', 'sent_id', and 'name'
#Also has 'labels', with relation label 'r', head 'h', tail 't'(head and tail are based on vertexSet), and 'evidence' a list of sentence ids which evidence the relation
#'title' of document
#'sents' list of sentences(pre-tokenized)
