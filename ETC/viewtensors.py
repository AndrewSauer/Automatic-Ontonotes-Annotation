import json,pickle
from etcmodel.models import tokenization
tokenizer=tokenization.FullTokenizer(
        vocab_file="./etcmodel/pretrained/etc_base/vocab_bert_uncased_english.txt",
        do_lower_case=True)
ENTITY_TYPES=['ORG','LOC','NUM','TIME','MISC','PER']
CONSTIT_TYPES=['ROOT', 'S', 'NP', 'NNP', 'NNPS', 'PRN', ',', 'VP', 'VBD', 'PP', 'IN', '-LRB-', 'RB', 'JJ', 'CC', '-RRB-', 'DT', 'NML', 'HYPH', 'NN', 'VBN', '.', 'PRP', 'ADJP', 'NNS', 'VBG', 'CD', 'PRP$', 'ADVP', 'TO', 'VB', 'QP', 'JJR', 'POS', 'VBZ', 'SBAR', 'VBP', 'EX', 'RBR', 'NP-TMP', 'WHNP', 'WDT', '``', "''", 'JJS', 'WP', 'SINV', 'FRAG', 'WHADVP', 'WRB', 'WHPP', 'PRT', 'RP', 'UCP', 'WP$', ':', 'FW', 'RRC', 'SYM', '$', 'CONJP', 'AFX', 'SBARQ', 'SQ', 'RBS', 'NAC', 'NFP', 'MD', 'PDT', 'UH', 'INTJ', 'ADD', 'NX', 'GW', 'X', 'LST', 'LS', 'WHADJP']
SRL_ARGUMENTS=['V','A0','A1','A2','A3','A4','A5','LOC','ADV','TMP','MNR','PRD','CAU','DIS','GOL','COM','EXT','PRP','DIR','NEG','PNC','MOD','REC','ADJ','LVB']
WSD_TYPES=pickle.load(open("senses.pickle",'rb'))
SRL_TYPES=pickle.load(open("frames.pickle",'rb'))

example_num=26
ANNOTATE=True
CONSTITFULL=False
TRUEFULL=True
DEV=False

filename="BERT"
if ANNOTATE:
    if TRUEFULL:
        filename+="truefull"
    else:
        filename+="anno"
        if CONSTITFULL:
            filename+="full"
if DEV:
    filename+="dev"
filename+="examples/tf_example"+str(example_num)+".pickle"
iotensors=pickle.load(open(filename,'rb'))
if DEV:
    docred=json.load(open("../docred_data/dev.json"))
    onf=json.load(open("../DocREDdev/dev"+str(example_num)+".final.json"))
else:
    docred=json.load(open("../docred_data/train_annotated.json"))
    onf=json.load(open("../DocREDtrain/train_annotated"+str(example_num)+".final.json"))
doc=docred[example_num]
features=iotensors[0]
labels=iotensors[1]
tokid=features['token_ids'][0].numpy().tolist()
tok=tokenizer.convert_ids_to_tokens(tokid)
gltok=features['global_token_ids'][0].numpy().tolist()
l2l=features['l2l_relative_att_ids'][0].numpy().tolist()
l2g=features['l2g_relative_att_ids'][0].numpy().tolist()
g2l=features['g2l_relative_att_ids'][0].numpy().tolist()
g2g=features['g2g_relative_att_ids'][0].numpy().tolist()
l2lmask=features['l2l_att_mask'][0].numpy().tolist()
l2gmask=features['l2g_att_mask'][0].numpy().tolist()
g2lmask=features['g2l_att_mask'][0].numpy().tolist()
g2gmask=features['g2g_att_mask'][0].numpy().tolist()
for i in range(len(g2l)):
    for j in range(len(g2l[i])):
        if g2l[i][j]!=l2g[j][i] and not (g2l[i][j]==0 and l2g[j][i]==1):
            print("g2l and l2g don't match! "+str(i)+" "+str(j))
        l2gcondition=gltok[i]*tokid[j]!=0 or gltok[i]+tokid[j]==0
        g2lcondition=l2gcondition and g2l[i][j]!=0
        if l2gcondition!=(l2gmask[j][i]==1) and gltok[i]+tokid[j]>0:
            print("l2g mask error: "+str(j)+" "+str(i))
        if g2lcondition!=(g2lmask[i][j]==1):
            print("g2l mask error: "+str(i)+str(j))
    for j in range(len(g2g[i])):
        if g2g[i][j]==26 and not g2g[j][i]==27:
            print("g2g 26/27 error! "+str(i)+str(j))
        g2gcondition=gltok[i]*gltok[j]!=0 or gltok[i]+gltok[j]==0
        if g2gcondition!=(g2gmask[i][j]==1):
            print("g2g mask error: "+str(i)+str(j))
print(l2l[0])
for i in range(len(l2l)):
    if l2l[i]!=l2l[0]:
        print(i,end=' ')
    for j in range(len(l2l[i])):
        if i+j>=84 and i+j-84<len(tokid):
            if tokid[i+j-84]*tokid[i]>0 or tokid[i+j-84]+tokid[i]==0:
                l2lcondition=True
            else:
                l2lcondition=False
        else:
            l2lcondition=False
        if l2lcondition!=l2lmask[i][j]:
            print("l2l mask error: "+str(i)+str(j))
def printtok(word):
    if word[0:2]=="##":
        print(word[2:],end='')
    else:
        print(" "+word,end='')
for i in range(len(onf['toktext'])):
    for word in onf['toktext'][i]:
        print(word[0],end=' ')
    print('\n')
    if not gltok[i]==1:
        print("gltok error: "+str(i))
    for j in range(len(g2l[i])):
        if g2l[i][j]==2:
            printtok(tok[j])
    print('\n')
for i in range(len(doc['vertexSet'])):
    for mention in doc['vertexSet'][i]:
        print(mention['type'],end=' ')
        for j in range(mention['pos'][0],mention['pos'][1]):
            print(doc['sents'][mention['sent_id']][j],end=' ')
    print('\n')
    print(ENTITY_TYPES[gltok[30+i]-4])
    for j in range(len(g2l[30+i])):
        if g2l[i+30][j]==3:
            printtok(tok[j])
    for j in range(len(g2g[30+i])):
        if g2g[30+i][j]==26:
            print(" "+str(j),end='')
    print('\n')
if not ANNOTATE:
    exit()
for i in range(len(onf['coref'])):
    for mention in onf['coref'][i]:
        for j in range(mention[1],mention[2]+1):
            print(onf['toktext'][mention[0]][j][0],end=' ')
    print('\n')
    if not gltok[i+80]==10:
        print("gltok error: "+str(i))
    for j in range(len(g2l[80+i])):
        if g2l[i+80][j]==4:
            printtok(tok[j])
    for j in range(len(g2g[80+i])):
        if g2g[80+i][j]==26:
            print(" "+str(j),end='')
    print('\n')
counter=0
for i in range(len(onf['srl'])):#CONTINUE
    for frame in onf['srl'][i]['arguments']:
        print(onf['srl'][i]['plemma_ids'][onf['srl'][i]['arguments'].index(frame)],end=' ')
        for argument in frame:
            print(argument[2],end=' ')
            for j in range(argument[0],argument[1]+1):
                print(onf['toktext'][i][j][0],end=' ')
        print('\n')
        print(SRL_TYPES[gltok[counter+100]-22875],end=' ')
        for j in range(len(g2l[counter+100])):
            if g2l[counter+100][j]!=0:
                if j==0 or g2l[counter+100][j]!=g2l[counter+100][j-1]:
                    print(SRL_ARGUMENTS[g2l[counter+100][j]-5],end=' ')
                printtok(tok[j])
        for j in range(len(g2g[counter+100])):
            if g2g[counter+100][j]==26:
                print(" "+str(j),end='')
        print('\n')
        counter+=1
for i in range(len(onf['constit'])):
    print(onf['constit'][i])
if CONSTITFULL:
    upper=575
else:
    upper=360
for i in range(190,upper):
    if gltok[i]!=0:
        print(str(i),end=' ')
        print(CONSTIT_TYPES[gltok[i]-12],end=' ')
        for j in range(len(g2l[i])):
            if g2l[i][j]==30:
                printtok(tok[j])
        for j in range(len(g2g[i])):
            if g2g[i][j]==26:
                print(" "+str(j),end='')
        print('\n')
if not CONSTITFULL:
    if TRUEFULL:
        base=575
    else:
        base=360
    for i in range(len(onf['wsd'])):
        print(onf['wsd'][i][2]+" "+onf['toktext'][onf['wsd'][i][0]][onf['wsd'][i][1]][0])
        print(WSD_TYPES[gltok[i+base]-90],end=' ')
        for j in range(len(g2l[i+base])):
            if g2l[i+base][j]==31:
                printtok(tok[j])
        for j in range(len(g2g[base+i])):
            if g2g[base+i][j]==26:
                print(" "+str(j),end='')
        print('\n')
#
