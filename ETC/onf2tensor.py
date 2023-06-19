import tensorflow as tf;
import numpy as np;
import os;
import json;
import pickle;
from transformers import AutoTokenizer
from etcmodel import feature_utils
from etcmodel.models import tokenization
#import ../output2onf.py #for the OntonotesAnnotation object

#global variables for feature ablation and other settings

FULLCONSTIT=False#if we set constit to full, meaning all non-leaf nodes
WSD=True#Whether to include WSD info

COREF=True#true iff we are merging entity mentions into one
E_TYPE=True#true iff we put the entity types in global_token_ids

LOCAL_SIZE=800;
NUM_SENTENCES=30
if COREF:
    NUM_ENTITIES=50;
    NUM_LABELS=96
else:
    NUM_ENTITIES=90;#space in global tokens reserved for Entities
    NUM_LABELS=97
NUM_ONF_COREF=20#Number of onf coreference nodes
NUM_ONF_SRL=90#Number of onf relation nodes
if FULLCONSTIT:
    NUM_ONF_CONSTIT=385
else:
    NUM_ONF_CONSTIT=170#Number of constituency parsing nodes in global
if WSD:
    NUM_ONF_WSD=255#Number of word sense disambiguation nodes in global(REDUCE)
else:
    NUM_ONF_WSD=0
GLOBAL_SIZE=NUM_SENTENCES+NUM_ENTITIES
ONF_ADDITION=NUM_ONF_COREF+NUM_ONF_SRL+NUM_ONF_CONSTIT+NUM_ONF_WSD
ONF_GLOBAL_SIZE=NUM_SENTENCES+NUM_ENTITIES+ONF_ADDITION
#Initialize tokenizer
TOKENIZER="BERT"
if TOKENIZER=="ROBERTA-BASE":
    tokenizer=AutoTokenizer.from_pretrained("roberta-base");
    with open("etcmodel/pretrained/etc_config.json") as f:
        config=json.load(f);#for relative_pos_max_distance and local_radius

elif TOKENIZER=="ALBERT":
    tokenizer=tokenization.FullTokenizer(
            vocab_file=None,
            do_lower_case=None,
            spm_model_file="etcmodel/pretrained/vocab_gpt_for_etc.model")
    with open("etcmodel/pretrained/etc_config.json") as f:
        config=json.load(f);#for relative_pos_max_distance and local_radius

elif TOKENIZER=="BERT":
    tokenizer=tokenization.FullTokenizer(
            vocab_file="./etcmodel/pretrained/etc_base/vocab_bert_uncased_english.txt",
            do_lower_case=True)
    with open("etcmodel/pretrained/etc_base/etc_config.json") as f:
        config=json.load(f);#for relative_pos_max_distance and local_radius

else:
    print("Choose a tokenizer!")
    exit()

whitespace=" \t\n\v\f\r\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"#list of unicode whitespace characters I know about

#SRL_ARGUMENTS=['V', 'N-ARG-A1', 'N-ARG-A2', 'N-ARGM-LOC', 'N-ARG-A0', 'N-ARGM-ADV', 'N-ARGM-TMP', 'N-ARGM-MNR', 'N-ARGM-PRD', 'N-ARGM-CAU', 'N-ARG-A3', 'N-ARGM-DIS', 'N-ARGM-GOL', 'R-ARG-A1', 'N-ARGM-COM', 'N-ARGM-EXT', 'N-ARG-A4', 'C-ARG-A1', 'N-ARGM-PRP', 'R-ARGM-LOC', 'N-ARGM-DIR', 'N-ARGM-NEG', 'R-ARG-A0', 'R-ARG-A2', 'R-ARGM-COM', 'N-ARGM-PNC', 'R-ARGM-GOL', 'N-ARGM-MOD', 'R-ARG-A4', 'N-ARGM-REC', 'N-ARGM-ADJ', 'R-ARGM-TMP', 'C-ARG-A0', 'C-ARG-A2', 'C-ARGM-EXT', 'R-ARGM-CAU', 'R-ARGM-DIR', 'C-ARGM-ADV', 'N-ARGM-LVB', 'R-ARGM-MNR', 'C-ARGM-LOC', 'R-ARG-A3', 'C-ARGM-MNR', 'N-ARG-A5', 'R-ARGM-ADV', 'C-ARG-A4', 'R-ARGM-EXT','C-ARGM-CAU']
SRL_ARGUMENTS=['V','A0','A1','A2','A3','A4','A5','LOC','ADV','TMP','MNR','PRD','CAU','DIS','GOL','COM','EXT','PRP','DIR','NEG','PNC','MOD','REC','ADJ','LVB']
FRAMELIST=pickle.load(open("frames.pickle",'rb'))
SENSELIST=pickle.load(open("senses.pickle",'rb'))

print(5+len(SRL_ARGUMENTS)+1)
def srlNormalForm(s):#slim down the relative vocabulary by grouping them together into the original propbank arguments
    result=""
    for c in s:
        if c=='-':
            result=""
        else:
            result+=c
    return result

CONSTIT_TYPES=['ROOT', 'S', 'NP', 'NNP', 'NNPS', 'PRN', ',', 'VP', 'VBD', 'PP', 'IN', '-LRB-', 'RB', 'JJ', 'CC', '-RRB-', 'DT', 'NML', 'HYPH', 'NN', 'VBN', '.', 'PRP', 'ADJP', 'NNS', 'VBG', 'CD', 'PRP$', 'ADVP', 'TO', 'VB', 'QP', 'JJR', 'POS', 'VBZ', 'SBAR', 'VBP', 'EX', 'RBR', 'NP-TMP', 'WHNP', 'WDT', '``', "''", 'JJS', 'WP', 'SINV', 'FRAG', 'WHADVP', 'WRB', 'WHPP', 'PRT', 'RP', 'UCP', 'WP$', ':', 'FW', 'RRC', 'SYM', '$', 'CONJP', 'AFX', 'SBARQ', 'SQ', 'RBS', 'NAC', 'NFP', 'MD', 'PDT', 'UH', 'INTJ', 'ADD', 'NX', 'GW', 'X', 'LST', 'LS', 'WHADJP']

def match(document,phrase,position):#Take a regularized document and entity string, see if the string fits into the document at 'position', ignoring non-letter characters
    alphabet="qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    curpos=position
    for letter in phrase:
        if letter in alphabet:
            while curpos<len(document) and not (document[curpos] in alphabet):
                curpos+=1;
            if curpos==len(document) or letter!=document[curpos]:
                return 0;
            curpos+=1;
    return curpos-position;
#convert a token range from those in the json data file to those in the onf data file derived from it
#onftokens and jsontokens are lists of sentences which are themselves lists of tokens
def tokenconvert(onftokens,jsontokens,jsonrange):
    onfregular="";#regularized string for onftokens
    onfstarts=[0];#starts of each token, plus the length of the string at the end
    onfsents=[0];#starts of each sentence, plus the length of the string at the end
    entityregular="";#regularized string for the entity from the jsontokens and jsonrange
    cur=0;#current position in regularization
    for sentence in onftokens:
        for word in sentence:
            for c in word:
                if not (c in whitespace):#no whitespace
                    if (c>='A' and c<='Z'):#all lowercase
                        onfregular+=chr(ord(c)+32);
                    else:
                        onfregular+=c;
                    cur+=1;
            if cur!=onfstarts[len(onfstarts)-1]:
                onfstarts.append(cur);
        onfsents.append(cur);
    original=0;#record where in the regularized json the original entity is
    for i in range(jsonrange[2]):
        for j in range(len(jsontokens[i])):
            for c in jsontokens[i][j]:
                if not (c in whitespace):
                    original+=1;
    for i in range(jsonrange[0]):
        for c in jsontokens[jsonrange[2]][i]:
            if not (c in whitespace):
                original+=1;
    for i in range(jsonrange[0],jsonrange[1]):
        for c in jsontokens[jsonrange[2]][i]:
            if not (c in whitespace):
                if (c>='A' and c<='Z'):
                    entityregular+=chr(ord(c)+32);
                else:
                    entityregular+=c;
    x=-1;#current minimum deviation from original range
    regularrange=[-1,-1];
    for i in range(len(onfregular)-len(entityregular)+1):#find best position for regularrange(there may be multiple matching, find which one is closest to the original position
        #add condition where, if entityregular not in onfregular, then we can match them with match(document,phrase,position)
        if (onfregular[i:i+len(entityregular)]==entityregular and (abs(i-original)<x or x==-1)) or (not (entityregular in onfregular) and match(onfregular,entityregular,i)>0):
            x=abs(i-original);
            if entityregular in onfregular:
                regularrange=[i,i+len(entityregular)];
            else:#if not exact match, the length of the inexact match is indicated by match()
                regularrange=[i,i+match(onfregular,entityregular,i)]
    if x==-1:
        print("Tokenization error: entity: "+entityregular+" onf: "+onfregular);
    onfrange=[-1,-1,-1];#now find the smallest onf token range which includes that character range
    cursent=0;
    for i in range(len(onfstarts)-1):
        if onfstarts[i] in onfsents:#update sentence when our word starts it
            cursent=onfsents.index(onfstarts[i]);
        if onfstarts[i+1]>regularrange[0] and onfrange[0]==-1:
            onfrange[0]=i-onfstarts.index(onfsents[cursent]);#index of the token within the current sentence
            onfrange[2]=cursent;
        if onfstarts[i+1]>=regularrange[1] and onfrange[1]==-1:
            #we want the range to expand beyond the sentence if need be, so use the sentence offset we were at when the entity started
            #the entity belongs to the first sentence it appears in, since when entities appear in multiple sentences it is due to erroneous splitting on the entity itself
            onfrange[1]=i+1-onfstarts.index(onfsents[onfrange[2]]);
            if onfrange[2]!=cursent:#We will delete these error messages when it is confirmed that the desired behavior is achieved when entities span across sentences
                print("Warning: entity spans across onf sentences: "+str(onfrange[2])+" "+str(cursent));
                print("Entity: "+entityregular+" onf: "+onfregular);
    if onfrange[0]==-1 or onfrange[1]==-1 or onfrange[2]==-1:
        print("ONF token range error!");
    return onfrange;

#convert a range over large tokens to a range over subtokens
#you input the token_nums from the sentence you want plus those after...
def narrowtoken(token_nums,onfrange):
    start=0;
    end=0;
    for i in range(onfrange[0]):
        start+=token_nums[i];
    for i in range(onfrange[1]):#vertexSet "pos" is left-inclusive, right-exclusive just like Python ranges!
        end+=token_nums[i];
    return [start,end,onfrange[2]];

class ioTensors:
    def __init__(self,
            token_ids=None,
            global_token_ids=None,
            l2l_att_mask=None,
            g2g_att_mask=None,
            l2g_att_mask=None,
            g2l_att_mask=None,
            l2l_relative_att_ids=None,
            g2g_relative_att_ids=None,
            l2g_relative_att_ids=None,
            g2l_relative_att_ids=None,
            label_id=None,
            global_label_id=None
            ):
        #features/input
        self.token_ids=token_ids
        self.global_token_ids=global_token_ids
        self.l2l_att_mask=l2l_att_mask
        self.g2g_att_mask=g2g_att_mask
        self.l2g_att_mask=l2g_att_mask
        self.g2l_att_mask=g2l_att_mask
        self.l2l_relative_att_ids=l2l_relative_att_ids
        self.g2g_relative_att_ids=g2g_relative_att_ids
        self.l2g_relative_att_ids=l2g_relative_att_ids
        self.g2l_relative_att_ids=g2l_relative_att_ids

        self.label_id=label_id#labels/output
        self.global_label_id=global_label_id

def nodetypeid(s):#not complete, add more types
    nodetypes=["ROOT","S","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB",".",",","-LRB-","-RRB-","``","''"]
    if s in nodetypes:
        return nodetypes.index(s);
    else:
        print("ERROR: "+s+" is unrecognized tree node type!");
        return -1;

#take in the onf input file and json document detailing output, output example for just tokens, sentences and relative position
def onf2simple(onffile,jsondocument):
    with open(onffile,'r') as f:
        onflines=f.readlines();
    f.close();

    #list of string entity types
    types=['ORG','LOC','NUM','TIME','MISC','PER']
    entitytypes=[];#list of numbered type of each entity in the text
    #Format: [start token#, end token#, sentence#]
    #Also the tokens are the original tokens, not the subdivided ones
    entitylocs=[];#list of entity locations in the text
    entityids=[];#What is the index of this mention's coref chain?
    for i in range(len(jsondocument['vertexSet'])):
        for vertex in jsondocument['vertexSet'][i]:
            s=vertex['type'];
            if s in types:
                entitytypes.append(types.index(s));
            else:
                print("Fatal error: Unknown type!")
                exit()
            entitylocs.append([vertex['pos'][0],vertex['pos'][1],vertex['sent_id']])
            entityids.append(i);
    if ((not COREF) and len(entitylocs)>NUM_ENTITIES) or (COREF and len(jsondocument['vertexSet'])>NUM_ENTITIES):#we can't have too many entity mentions
        print("Fatal error! Too many entity mentions ("+str(len(entitylocs))+")");
        exit();
    glabel_update_values=[];#list of updates we make to global_label_id based on jsonfile
    glabel_update_indices=[];
    labelstrings=['P159','P17','P131','P150','P27','P569','P19','P172','P571','P576','P607','P30','P276','P1376','P206','P495','P551','P264','P527','P463','P175','P577','P161','P403','P20','P69','P570','P108','P166','P6','P361','P36','P26','P25','P22','P40','P37','P1412','P800','P178','P400','P937','P102','P585','P740','P3373','P1001','P57','P58','P272','P155','P156','P194','P241','P127','P118','P39','P674','P179','P1441','P170','P449','P86','P488','P1344','P580','P582','P676','P54','P50','P840','P136','P205','P706','P162','P710','P35','P140','P1336','P364','P737','P279','P31','P137','P112','P123','P176','P749','P355','P1198','P171','P1056','P1366','P1365','P807','P190']
    if COREF:#making the label tensor is simpler if we treat coreferring entities as the same
        for label in jsondocument['labels']:
            if not (label['r'] in labelstrings):
                print("Fatal error! Label not found!"+label['r'])
                exit()
            glabel_update_values.append(1)
            glabel_update_indices.append([0,label['h'],label['t'],labelstrings.index(label['r'])])
    else:
        for label in jsondocument['labels']:#add in the labels
            #add in relations for every pair of mentions with the proper id
            for i in range(len(entityids)):
                for j in range(len(entityids)):
                    if entityids[i]==label['h'] and entityids[j]==label['t']:
                        if not (label['r'] in labelstrings):
                            print("Fatal error! Label not found!"+label['r'])
                            exit();
                        #glabel_update_values.append(labelstrings.index(label['r'])+2);#label it based on the global list of labels
                        #glabel_update_indices.append([0,i,j]);
                        #CHANGED FOR MULTILABELS
                        glabel_update_values.append(1)
                        glabel_update_indices.append([0,i,j,labelstrings.index(label['r'])+1])
        for i in range(len(entityids)):#add in relations (denoted with 1) for each pair of mentions of the same entity
            for j in range(len(entityids)):
                if entityids[i]==entityids[j]:
                    #glabel_update_values.append(1);
                    #glabel_update_indices.append([0,i,j]);
                    #CHANGED FOR MULTILABELS
                    glabel_update_values.append(1)
                    glabel_update_indices.append([0,i,j,0])
    if len(glabel_update_indices)!=len(jsondocument['labels']):
        print("Fatal error! Mismatch in number of positive labels!")
        print(len(glabel_update_indices))
        print(len(jsondocument['labels']))
        exit()
    #split up tokens if they're not in vocab
    token_ids=[];
    token_nums=[];#how many RoBERTa tokens for each onf token?
    large_tokens=[];#the list of onf tokens as strings, separated by sentence
    global_token_ids=[];
    sentence_starts=[];#List of indexes at which the sentences start(the actual start token, not the 0
    mode="None"
    #read in all the word and sentence tokens
    for i in range(len(onflines)):
        if i<len(onflines)-1 and "Leaves:" in onflines[i] and "-------" in onflines[i+1]:
            mode="Words";
            if TOKENIZER=="ROBERTA-BASE":
                token_ids.append(0);
            large_tokens.append([]);
            token_nums.append([]);
            sentence_starts.append(len(token_ids))
        elif mode=="Words":
            if "--------------------------" in onflines[i] or "====================" in onflines[i] or i==len(onflines)-1:
                mode="None";
                global_token_ids.append(1);#1 means sentence in global_ids, since that is what the pretrained models were trained on
                if TOKENIZER=="ROBERTA-BASE":
                    token_ids.append(2);#2 at end of sentence
            else:
                tokens=[""];#first split the line into tokens, then check if it's introducing a new word
                for c in onflines[i]:
                    if c in whitespace and tokens[len(tokens)-1]!="":
                        tokens.append("");
                    elif not (c in whitespace):
                        tokens[len(tokens)-1]+=c;
                if tokens[len(tokens)-1]=="":
                    tokens.pop();
                #introduces new word if first token is number
                newWord=False;#obviously no new word if line is empty
                if len(tokens)>0:
                    newWord=True;
                    for c in tokens[0]:
                        if not (c in "0123456789"):
                            newWord=False;
                            break;
                if newWord:#combine all tokens after the first, then tokenize them
                    word="";
                    if len(tokens)>2:
                        print('"'+line+"\" appears to have too many words!")
                    for j in range(1,len(tokens)):
                        if j>1 or int(tokens[0])>0:#Put space before the word unless it is the first word in the sentence
                            word+=" ";
                        word+=tokens[j];
                    large_tokens[len(large_tokens)-1].append(word);#Add word to large_tokens so we can try to match it to the json tokens
                    if TOKENIZER=="ROBERTA-BASE":
                        word=tokenizer(word)["input_ids"];
                        word.remove(0)#remove start and end
                        word.remove(2)
                    elif TOKENIZER=="ALBERT" or TOKENIZER=="BERT":
                        word=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                    #add tokenized word to token_ids
                    token_ids.extend(word);
                    token_nums[len(token_nums)-1].append(len(word));
    if len(large_tokens)>GLOBAL_SIZE-NUM_ENTITIES:
        print("Fatal error! Too many sentences! ("+str(len(large_tokens))+")");
        exit();
    example=ioTensors();
    #construct token_ids
    example.token_ids=tf.reshape(tf.convert_to_tensor(token_ids),[1,len(token_ids)]);
    long_padding=tf.zeros([1,LOCAL_SIZE-len(token_ids)],tf.int32);
    example.token_ids=tf.concat([example.token_ids,long_padding],1);
    #construct global_token_ids
    #1 means sentence, 4+ means entity, 0 means nothing
    #For this we only give info on where the entities are
    example.global_token_ids=tf.reshape(tf.convert_to_tensor(global_token_ids),[1,len(global_token_ids)]);
    global_padding=tf.zeros([1,GLOBAL_SIZE-NUM_ENTITIES-len(global_token_ids)],tf.int32);
    #set the number of actual, non-padding entities we're putting in
    if COREF:
        actual_entities=len(jsondocument['vertexSet'])
    else:
        actual_entities=len(entitylocs)
    entities=4*tf.ones([1,actual_entities],tf.int32)
    entity_padding=tf.zeros([1,NUM_ENTITIES-actual_entities],tf.int32)
    example.global_token_ids=tf.concat([example.global_token_ids,global_padding,entities,entity_padding],1);
    global_token_ids_segments=example.global_token_ids#we've defined the segments
    if E_TYPE:#If we're distinguishing entity types in global_token_ids, we do it here
        updates=[]
        indices=[]
        for i in range(len(entityids)):
            if not COREF:
                updates.append(4+entitytypes[i])
                indices.append([0,i+NUM_SENTENCES])
            elif i==0 or entityids[i]!=entityids[i-1]:#when using coref, we only add in the type of the first entity mention
                updates.append(4+entitytypes[i])
                indices.append([0,entityids[i]+NUM_SENTENCES])
        example.global_token_ids=tf.tensor_scatter_nd_update(example.global_token_ids,tf.constant(indices),tf.constant(updates))

    #label_id is currently completely meaningless, don't use it
    example.label_id=example.token_ids;
    #example.global_label_id=tf.zeros([1,NUM_ENTITIES,NUM_ENTITIES],tf.int32)
    #example.global_label_id=tf.tensor_scatter_nd_update(example.global_label_id,tf.constant(glabel_update_indices),tf.constant(glabel_update_values))
    #CHANGED FOR MULTILABELS
    example.global_label_id=tf.zeros([1,NUM_ENTITIES,NUM_ENTITIES,NUM_LABELS],tf.int32)
    #apparently some documents have no labels! good to know I guess
    if len(glabel_update_indices)>0:
        example.global_label_id=tf.tensor_scatter_nd_update(example.global_label_id,tf.constant(glabel_update_indices),tf.constant(glabel_update_values))

    max_distance=config["relative_pos_max_distance"];
    relativePositionGenerator=feature_utils.RelativePositionGenerator(max_distance);
    #relative position attention for local tokens
    example.l2l_relative_att_ids=relativePositionGenerator.make_local_relative_att_ids(
            seq_len=LOCAL_SIZE,
            local_radius=config["local_radius"])#QUESTION: Why would the attention distance be different from the max relative position distance?
            #ANSWER: relative position measurement increases the vocabulary
    #relative position attention for sentences
    #CONTINUE: Remove the relative position info for entities and make sure it doesn't impact performance
    example.g2g_relative_att_ids=relativePositionGenerator.make_relative_att_ids(
            seq_len=GLOBAL_SIZE)
    #overwrite relative attention between sentences and entities
    example.g2g_relative_att_ids=feature_utils.overwrite_relative_att_ids_outside_segments(
            rel_att_ids=example.g2g_relative_att_ids,
            segment_ids=global_token_ids_segments,
            overwrite_value=max_distance*2+1)
    #max_distance*2+1: overwritevalue
    #max_distance*2+2: entity in sentence
    #max_distance*2+3: sentence contains entity
    updates=[];
    indices=[];
    for i in range(len(entitylocs)):
        if COREF:
            entity_index=GLOBAL_SIZE-NUM_ENTITIES+entityids[i]
        else:
            entity_index=GLOBAL_SIZE-NUM_ENTITIES+i
        updates.append(max_distance*2+2);
        indices.append([0,entity_index,entitylocs[i][2]]);
        updates.append(max_distance*2+3);
        indices.append([0,entitylocs[i][2],entity_index]);
    #Add in attention for which sentences contain which entities
    example.g2g_relative_att_ids=tf.tensor_scatter_nd_update(example.g2g_relative_att_ids,tf.constant(indices),tf.constant(updates));

    #simple ids: 1 means no relation, 2 means part of the sentence
    #3 means part of the entity
    example.g2l_relative_att_ids=tf.zeros([1,GLOBAL_SIZE,LOCAL_SIZE],tf.int32);#simple ids: 1 means not part of the sentence, 2 means part of the sentence
    example.l2g_relative_att_ids=tf.ones([1,LOCAL_SIZE,GLOBAL_SIZE],tf.int32);
    g2l_update_indices=[];#indices which should be 2 or 3
    l2g_update_indices=[];
    update_values=[];#all 2s and 3s, same length as above
    sentence=0;#sentence number we're on
    #cur_tok=0;#large onf token we're on
    #next_tok_index=token_nums[0][0];#small token index beginning next token
    #print("json sentences: "+str(jsondocument['sents'])+" onf sentences: "+str(large_tokens))
    for i in range(len(entitylocs)):#convert entity locations from json token range to onf token range
        entitylocs[i]=tokenconvert(large_tokens,jsondocument['sents'],entitylocs[i]);
        #now that the entity location can expand beyond the bounds of a sentence, we need to add all token_nums in or after the target sentence
        more_token_nums=token_nums[entitylocs[i][2]];
        for j in range(entitylocs[i][2]+1,len(token_nums)):
            more_token_nums.extend(token_nums[j]);
        entitylocs[i]=narrowtoken(more_token_nums,entitylocs[i]);
    for i in range(len(token_ids)):#find which tokens belong to which sentences
        if (TOKENIZER=="ALBERT" or TOKENIZER=="BERT") and i in sentence_starts:
            sentence=sentence_starts.index(i)
        g2l_update_indices.append([0,sentence,i]);
        l2g_update_indices.append([0,i,sentence]);
        update_values.append(2);
        if TOKENIZER=="ROBERTA-BASE" and token_ids[i]==2:
            sentence+=1;
        if (TOKENIZER=="ROBERTA-BASE" and i>0 and token_ids[i-1]==0) or ((TOKENIZER=="ALBERT" or TOKENIZER=="BERT") and i in sentence_starts):#on new sentence, add all entity connections from that sentence
            for x in range(len(entitylocs)):
                entity=entitylocs[x];
                if entity[2]==sentence:
                    #add an offset to j so it can continue into the next sentence, skipping over 0 and 2 if need be
                    offset=0;
                    for j in range(i+entity[0],i+entity[1]):
                        update_values.append(3);
                        while(TOKENIZER=="ROBERTA-BASE" and (token_ids[j+offset]==0 or token_ids[j+offset]==2)):
                            offset+=1;
                        if COREF:
                            entity_index=GLOBAL_SIZE-NUM_ENTITIES+entityids[x]
                        else:
                            entity_index=GLOBAL_SIZE-NUM_ENTITIES+x
                        g2l_update_indices.append([0,entity_index,j+offset]);
                        l2g_update_indices.append([0,j+offset,entity_index]);
    #update the attention ids to reflect which tokens are in which sentences
    example.g2l_relative_att_ids=tf.tensor_scatter_nd_update(example.g2l_relative_att_ids,tf.constant(g2l_update_indices),tf.constant(update_values));
    example.l2g_relative_att_ids=tf.tensor_scatter_nd_update(example.l2g_relative_att_ids,tf.constant(l2g_update_indices),tf.constant(update_values));

    #breakpoints at end of sentences, marked with 2 in token_ids
    #att_breakpoints=tf.zeros_like(token_ids)
    #for i in range(tf.size(token_ids)):
    #    if token_ids[i]==2:
    #        att_breakpoints[i]=1;
    #QUESTION: do we really want to mask cross-sentence attention?
    #l2l_att_mask=feature_utils.make_local_att_mask_from_breakpoints(
    #        att_breakpoints=att_breakpoints,
    #        local_radius=maxDistance)
    #make attention masks which screen off the padding from consideration in l2l and g2g
    long_mask=tf.concat([tf.ones([1,len(token_ids)],tf.int32),tf.zeros([1,LOCAL_SIZE-len(token_ids)],tf.int32)],1)
    global_mask=tf.concat([tf.ones([1,len(global_token_ids)],tf.int32),tf.zeros([1,GLOBAL_SIZE-NUM_ENTITIES-len(global_token_ids)],tf.int32),tf.ones([1,actual_entities],tf.int32),tf.zeros([1,NUM_ENTITIES-actual_entities],tf.int32)],1)
    example.l2l_att_mask=feature_utils.make_local_segmented_att_mask(
            segment_ids=long_mask,
            local_radius=config["local_radius"])
    example.g2g_att_mask=feature_utils.make_segmented_att_mask(segment_ids=global_mask);

    #mask everything that involves the padding
    example.l2g_att_mask=tf.zeros([1,LOCAL_SIZE,GLOBAL_SIZE],tf.int32);
    example.g2l_att_mask=tf.zeros([1,GLOBAL_SIZE,LOCAL_SIZE],tf.int32);
    g2l_update_indices=[];
    l2g_update_indices=[];
    update_values=[];
    for i in range(LOCAL_SIZE):
        for j in range(GLOBAL_SIZE):
            if i<len(token_ids) and (j<len(global_token_ids) or (j>=GLOBAL_SIZE-NUM_ENTITIES and j<GLOBAL_SIZE-NUM_ENTITIES+actual_entities)):
                g2l_update_indices.append([0,j,i]);
                l2g_update_indices.append([0,i,j]);
                update_values.append(1);
    example.l2g_att_mask=tf.tensor_scatter_nd_update(example.l2g_att_mask,tf.constant(l2g_update_indices),tf.constant(update_values));
    #example.g2l_att_mask=tf.tensor_scatter_nd_update(example.g2l_att_mask,tf.constant(g2l_update_indices),tf.constant(update_values));
    #In the paper it shows a diagram, g2l should be masked except where there is a relation.
    example.g2l_att_mask=tf.cast(tf.greater(example.g2l_relative_att_ids,example.g2l_att_mask),"int32")
    return example,token_nums,sentence_starts,(len(token_ids),len(global_token_ids),actual_entities);
#take in a tuple (sentence#,starttok#,endtok#) for onf tokens, and convert it to an long_index range
#starttok and endtok should be a python-style range, inclusive at beginning, exclusive at end
def tuple2index(token_ids,token_nums,sentence_starts,t):
    if TOKENIZER=="ALBERT" or TOKENIZER=="BERT":
        start=sentence_starts[t[0]]
    elif TOKENIZER=="ROBERTA-BASE":
        sentence=-1
        start=0
        for i in range(len(token_ids)):
            if token_ids[i]==0:
                sentence+=1
            if sentence==t[0]:
                start=i+1
                break
    else:
        print("Pick a tokenizer!")
        exit()
    for i in range(t[1]):#go to the start of our tuple
        start+=token_nums[t[0]][i]
    end=start
    for i in range(t[1],t[2]):#go to the end of our tuple
        end+=token_nums[t[0]][i]
    return (start,end)

#Take in a list of strings representing a constituency parses, return a list of lists of nodes and relations representing the tree structure
def constit2tree(constit,sents):
    #split into tokens simply with whitespace, and assuming all parenthesis are individual tokens
    constittoks=[]
    for s in constit:
        constittoks.append([""])
        for c in s:
            if c in whitespace and constittoks[-1][-1]!="":
                constittoks[-1].append("")
            elif c=='(' or c==')':
                if constittoks[-1][-1]=="":
                    constittoks[-1][-1]=c
                else:
                    constittoks[-1].append(c)
                constittoks[-1].append("")
            elif not c in whitespace:
                constittoks[-1][-1]+=c
        if constittoks[-1][-1]=="":
            constittoks[-1].pop()
    if len(constittoks)!=len(sents):
        print("Fatal error: Mismatch in number of sentences!")
        print(constittoks)
        print(sents)
        exit()
    nodes=[]
    ranges=[]
    relations=[]
    for i in range(len(constittoks)):
        nodes.append([])
        ranges.append([])#onf token ranges corresponding to each node
        relations.append([])#each entry (a,b) means that a is below b in the tree structure
        abovenodes=[]#indices of each node above our current position in the tree
        token=0#id of next onf token
        for j in range(len(constittoks[i])):
            if j>0 and constittoks[i][j-1]=='(':
                if constittoks[i][j] in CONSTIT_TYPES:
                    if constittoks[i][j][0]=='N' or FULLCONSTIT==True:
                        nodes[-1].append(CONSTIT_TYPES.index(constittoks[i][j]))
                        ranges[-1].append([i,token])#we add the third value in the range later when this node is popped from the abovenodes stack
                        abovenodes.append(len(nodes[-1])-1)
                    else:
                        abovenodes.append(-1)
                else:
                    print("Fatal error: Unrecognized constituency parse node! "+constittoks[i][j])
                    print(constittoks)
                    print(sents)
                    exit()
            elif constittoks[i][j]==')':
                poppednode=abovenodes.pop()
                if poppednode!=-1:
                    ranges[-1][poppednode].append(token)
                    #If the range is only one, remove the node
                    if ranges[-1][poppednode][2]-ranges[-1][poppednode][1]==1 and poppednode==len(ranges[-1])-1:
                        nodes[-1].pop()
                        ranges[-1].pop()
                    else:#otherwise we keep it and add in its relations
                        for node in abovenodes:
                            if node!=-1:
                                relations[-1].append((poppednode,node))
            elif constittoks[i][j]!='(':
                token+=1
        if token!=len(sents[i]):
            print("Fatal error: Mismatch in number of tokens in sentence "+str(i)+"!")
            print(token)
            print(len(sents[i]))
            exit()
    return nodes,ranges,relations

"""x=0
for i in range(3053):
    print(i)
    jsonobject=json.load(open("../DocREDtrain/train_annotated"+str(i)+".final.json"))
    nodes,ranges,relations=constit2tree(jsonobject['constit'],jsonobject['toktext'])
    if i==3052:
        print(nodes)
        print(ranges)
        print(relations)
    size=0
    for sent in nodes:
        size+=len(sent)
    if size>x:
        x=size
print(x)
exit()"""

#take in the onf input file and basic tensors from onf2simple, modify the tensors to include more detailed annotation info
def onf2tensor(onffile,simple,token_nums,sentence_starts,actual_lengths):
    onfdict=json.load(open(onffile))
    result=simple;
    max_distance=config["relative_pos_max_distance"]
    #How many of these are actually in the document rather than being padding
    actual_long_tokens=actual_lengths[0]
    actual_sentences=actual_lengths[1]
    actual_entities=actual_lengths[2]
    #pad out the tensors to make space for the new data
    result.global_token_ids=tf.pad(simple.global_token_ids,tf.constant([[0,0],[0,ONF_ADDITION]]))
    #result.g2g_relative_att_ids=tf.pad(simple.g2g_relative_att_ids,tf.constant([[0,0],[0,ONF_ADDITION],[0,ONF_ADDITION]]),constant_values=max_distance*2+1)
    result.g2l_relative_att_ids=tf.pad(simple.g2l_relative_att_ids,tf.constant([[0,0],[0,ONF_ADDITION],[0,0]]),)
    result.l2g_relative_att_ids=tf.pad(simple.l2g_relative_att_ids,tf.constant([[0,0],[0,0],[0,ONF_ADDITION]]),constant_values=1)

#    result.g2g_att_mask=tf.pad(simple.g2g_att_mask,tf.constant([[0,0],[0,ONF_ADDITION],[0,ONF_ADDITION]]))
#    result.g2l_att_mask=tf.pad(simple.g2l_att_mask,tf.constant([[0,0],[0,ONF_ADDITION],[0,0]]))
#    result.l2g_att_mask=tf.pad(simple.l2g_att_mask,tf.constant([[0,0],[0,0],[0,ONF_ADDITION]]))

#    global_segments=tf.multiply(global_mask,tf.concat([tf.ones([1,NUM_SENTENCES],tf.int32),2*tf.ones([1,NUM_ENTITIES],tf.int32),3*tf.ones([1,NUM_ONF_COREF],tf.int32),4*tf.ones([1,NUM_ONF_SRL],tf.int32),5*tf.ones([1,NUM_ONF_CONSTIT],tf.int32)],-1)
#for g2g, we simply have relations between sentences and the nodes within them.
#There's no reason to have relative_position info between global tokens other than sentences. We probably shouldn't have it for entities anyway.
    gltok_indices=[]
    gltok_updates=[]
    g2g_indices=[]
    g2g_updates=[]
    g2l_indices=[]
    l2g_indices=[]
    l2g2l_updates=[]
    token_ids_list=simple.token_ids.numpy().tolist()
    if len(onfdict['coref'])>NUM_ONF_COREF:
        print("Fatal error! Not enough space for "+str(len(onfdict['coref']))+" coreference nodes!")
        exit()
    actual_coref=len(onfdict['coref'])
    for i in range(len(onfdict['coref'])):#2d array of coref chains, each entry is tuple of (sentence#,starttok#,endtok#
        chain=onfdict['coref'][i]
        gltok_indices.append([0,GLOBAL_SIZE+i])
        gltok_updates.append(10)#10 is code for coref, coref info comes right after the entity info
        for j in range(len(chain)):
            mention=chain[j]
            sentence_num=mention[2]
            mention=tuple2index(token_ids_list,token_nums,sentence_starts,(mention[0],mention[1],mention[2]+1))
            g2g_updates.append(max_distance*2+2)
            g2g_indices.append([0,GLOBAL_SIZE+i,sentence_num])
            g2g_updates.append(max_distance*2+3)
            g2g_indices.append([0,sentence_num,GLOBAL_SIZE+i])
            for k in range(mention[0],mention[1]):
                g2l_indices.append([0,GLOBAL_SIZE+i,k])
                l2g_indices.append([0,k,GLOBAL_SIZE+i])
                l2g2l_updates.append(4)#4 means part of coref
    srl_index=GLOBAL_SIZE+NUM_ONF_COREF#srl comes right after coref
    framelist=FRAMELIST
    for i in range(len(onfdict['srl'])):#propbank propositions (same structure as JSON srl)
        #i is the id of the sentence we're looking at
        for j in range(len(onfdict['srl'][i]['arguments'])):
            if srl_index-GLOBAL_SIZE-NUM_ONF_COREF>=NUM_ONF_SRL:
                print("Fatal error! Not enough space for srl nodes!")
                exit()
            if onfdict['srl'][i]['plemma_ids'][j] not in framelist:
                print("Fatal error! Unrecognized frame!")
                exit()
            gltok_indices.append([0,srl_index])
            #gltok_updates.append(11)
            gltok_updates.append(12+len(CONSTIT_TYPES)+len(SENSELIST)+framelist.index(onfdict['srl'][i]['plemma_ids'][j]))
            #11 was code for srl. code is now determined by position in framelist
            g2g_updates.append(max_distance*2+2)
            g2g_indices.append([0,srl_index,i])
            g2g_updates.append(max_distance*2+3)
            g2g_indices.append([0,i,srl_index])
            frameinfo=onfdict['srl'][i]['arguments'][j]
            for argument in frameinfo:#adding argument info to l2g/g2l
                if not srlNormalForm(argument[2]) in SRL_ARGUMENTS:
                    print("Fatal error! "+srlNormalForm(argument[2])+" not in SRL_ARGUMENTS!")
                    exit()
                argumentrange=tuple2index(token_ids_list,token_nums,sentence_starts,(i,argument[0],argument[1]+1))
                for k in range(argumentrange[0],argumentrange[1]):
                    g2l_indices.append([0,srl_index,k])
                    l2g_indices.append([0,k,srl_index])
                    l2g2l_updates.append(5+SRL_ARGUMENTS.index(srlNormalForm(argument[2])))#separate id for each argument type
            srl_index+=1
    actual_srl=srl_index-GLOBAL_SIZE-NUM_ONF_COREF
    nodes,ranges,relations=constit2tree(onfdict['constit'],onfdict['toktext'])
    constit_index=GLOBAL_SIZE+NUM_ONF_COREF+NUM_ONF_SRL
    for i in range(len(nodes)):#Add nodes from the parse tree
        constit_base=constit_index#start of the nodes for this sentence
        for j in range(len(nodes[i])):
            if constit_index-GLOBAL_SIZE-NUM_ONF_COREF-NUM_ONF_SRL>=NUM_ONF_CONSTIT:
                print("Fatal error! Not enough space for constit nodes!")
                exit()
            gltok_indices.append([0,constit_index])
            gltok_updates.append(12+nodes[i][j])
            g2g_updates.append(max_distance*2+2)#relate the nodes to their sentences
            g2g_indices.append([0,constit_index,i])
            g2g_updates.append(max_distance*2+3)
            g2g_indices.append([0,i,constit_index])
            constit_index+=1
        for rel in relations[i]:#relate the nodes to each other
            g2g_updates.append(max_distance*2+2)
            g2g_indices.append([0,constit_base+rel[0],constit_base+rel[1]])
            g2g_updates.append(max_distance*2+3)
            g2g_indices.append([0,constit_base+rel[1],constit_base+rel[0]])
        for j in range(len(ranges[i])):#relate the nodes to the tokens in their spans
            narrowrange=tuple2index(token_ids_list,token_nums,sentence_starts,ranges[i][j])
            for k in range(narrowrange[0],narrowrange[1]):
                g2l_indices.append([0,constit_base+j,k])
                l2g_indices.append([0,k,constit_base+j])
                l2g2l_updates.append(5+len(SRL_ARGUMENTS))
    actual_constit=constit_index-GLOBAL_SIZE-NUM_ONF_COREF-NUM_ONF_SRL
    wsd=onfdict['wsd']
    senselist=SENSELIST
    if NUM_ONF_WSD>0:#allows us to easily remove this
        if len(wsd)>NUM_ONF_WSD:
            print("Fatal error! Not enough space for wsd nodes!")
            exit()
        for i in range(len(wsd)):
            sent=wsd[i][0]
            word=wsd[i][1]
            sense=wsd[i][2]
            if not sense in senselist:
                print("Fatal error: unrecognized word sense!")
                exit()
            narrowrange=tuple2index(token_ids_list,token_nums,sentence_starts,(sent,word,word+1))
            wsdindex=ONF_GLOBAL_SIZE-NUM_ONF_WSD+i
            gltok_indices.append([0,ONF_GLOBAL_SIZE-NUM_ONF_WSD+i])
            gltok_updates.append(12+len(CONSTIT_TYPES)+senselist.index(sense))
            g2g_updates.append(max_distance*2+2)#relate the nodes to their sentences
            g2g_indices.append([0,wsdindex,sent])
            g2g_updates.append(max_distance*2+3)
            g2g_indices.append([0,sent,wsdindex])
            for j in range(narrowrange[0],narrowrange[1]):
                g2l_indices.append([0,wsdindex,j])
                l2g_indices.append([0,j,wsdindex])
                l2g2l_updates.append(5+len(SRL_ARGUMENTS)+1)
        actual_wsd=len(wsd)
    else:
        actual_wsd=0

    result.global_token_ids=tf.tensor_scatter_nd_update(result.global_token_ids,tf.constant(gltok_indices),tf.constant(gltok_updates))
    global_mask=tf.cast(tf.greater(result.global_token_ids,tf.zeros(tf.shape(result.global_token_ids),tf.int32)),tf.int32)
    global_segments=tf.concat([tf.ones([1,NUM_SENTENCES],tf.int32),2*tf.ones([1,NUM_ENTITIES],tf.int32),3*tf.ones([1,NUM_ONF_COREF],tf.int32),4*tf.ones([1,NUM_ONF_SRL],tf.int32),5*tf.ones([1,NUM_ONF_CONSTIT],tf.int32)],axis=1)
    if NUM_ONF_WSD>0:
        global_segments=tf.concat([global_segments,6*tf.ones([1,NUM_ONF_WSD],tf.int32)],axis=1)
    global_segments=tf.multiply(global_segments,global_mask)
    relativePositionGenerator=feature_utils.RelativePositionGenerator(max_distance);
    tmp_g2g_relative_att_ids=relativePositionGenerator.make_relative_att_ids(
            seq_len=ONF_GLOBAL_SIZE)
    #overwrite relative attention between different annotation types
    tmp_g2g_relative_att_ids=feature_utils.overwrite_relative_att_ids_outside_segments(
            rel_att_ids=tmp_g2g_relative_att_ids,
            segment_ids=global_segments,
            overwrite_value=max_distance*2+1)
    #add only the outer part of this to our result, we can do this instead of padding
    result.g2g_relative_att_ids=tf.concat([tf.concat([simple.g2g_relative_att_ids,tmp_g2g_relative_att_ids[:,:GLOBAL_SIZE,GLOBAL_SIZE:]],axis=2),tmp_g2g_relative_att_ids[:,GLOBAL_SIZE:,:]],axis=1)

    result.g2g_relative_att_ids=tf.tensor_scatter_nd_update(result.g2g_relative_att_ids,tf.constant(g2g_indices),tf.constant(g2g_updates))
    result.g2l_relative_att_ids=tf.tensor_scatter_nd_update(result.g2l_relative_att_ids,tf.constant(g2l_indices),tf.constant(l2g2l_updates))
    result.l2g_relative_att_ids=tf.tensor_scatter_nd_update(result.l2g_relative_att_ids,tf.constant(l2g_indices),tf.constant(l2g2l_updates))
    result.g2g_att_mask=feature_utils.make_segmented_att_mask(segment_ids=global_mask);
    #mask everything that involves the padding
    result.l2g_att_mask=tf.zeros([1,LOCAL_SIZE,ONF_GLOBAL_SIZE],tf.int32);
    result.g2l_att_mask=tf.zeros([1,ONF_GLOBAL_SIZE,LOCAL_SIZE],tf.int32);
    l2g_update_indices=[];
    update_values=[];
    for i in range(LOCAL_SIZE):
        for j in range(ONF_GLOBAL_SIZE):
            #this conditional determines whether [0,i,j] is a relation between two actual tokens rather than padding
            if i<actual_long_tokens and (j<actual_sentences or (j>=NUM_SENTENCES and j<NUM_SENTENCES+actual_entities) or (j>=GLOBAL_SIZE and j<GLOBAL_SIZE+actual_coref) or (j>=GLOBAL_SIZE+NUM_ONF_COREF and j<srl_index) or (j>=GLOBAL_SIZE+NUM_ONF_COREF+NUM_ONF_SRL and j<constit_index) or (j>=ONF_GLOBAL_SIZE-NUM_ONF_WSD) and j<ONF_GLOBAL_SIZE-NUM_ONF_WSD+actual_wsd):
                l2g_update_indices.append([0,i,j]);
                update_values.append(1);
    result.l2g_att_mask=tf.tensor_scatter_nd_update(result.l2g_att_mask,tf.constant(l2g_update_indices),tf.constant(update_values));
    #In the paper it shows a diagram, g2l should be masked except where there is a relation.
    result.g2l_att_mask=tf.cast(tf.greater(result.g2l_relative_att_ids,result.g2l_att_mask),"int32")
    return result
#load the training/dev json file
jsontrainobject=json.load(open("../docred_data/train_annotated.json"))
jsondevobject=json.load(open("../docred_data/dev.json"))
jsonREtrainobject=json.load(open("../docred_data/train_revised.json"))
jsonREdevobject=json.load(open("../docred_data/dev_revised.json"))


#write examples
def writeexamples(jsonobject,inpath,outpath,annotate=True):
    for i in range(len(jsonobject)):
        if os.path.exists(outpath+str(i)+".pickle"):
            os.remove(outpath+str(i)+".pickle");
        print(i);
        iotensor,token_nums,sentence_starts,actual_lengths=onf2simple(inpath+str(i)+".onf",jsonobject[i]);
        features={
                "token_ids":iotensor.token_ids,
                "global_token_ids":iotensor.global_token_ids,
                "l2l_att_mask":iotensor.l2l_att_mask,
                "l2g_att_mask":iotensor.l2g_att_mask,
                "g2l_att_mask":iotensor.g2l_att_mask,
                "g2g_att_mask":iotensor.g2g_att_mask,
                "l2l_relative_att_ids":iotensor.l2l_relative_att_ids,
                "l2g_relative_att_ids":iotensor.l2g_relative_att_ids,
                "g2l_relative_att_ids":iotensor.g2l_relative_att_ids,
                "g2g_relative_att_ids":iotensor.g2g_relative_att_ids
        }
        labels={
                "label_id":iotensor.label_id,
                "global_label_id":iotensor.global_label_id
        }
        with open(TOKENIZER+outpath+str(i)+".pickle",'wb') as f:
            pickle.dump((features,labels),f);
        if not annotate:
            return
        iotensor=onf2tensor(inpath+str(i)+".final.json",iotensor,token_nums,sentence_starts,actual_lengths)
        features={
                "token_ids":iotensor.token_ids,
                "global_token_ids":iotensor.global_token_ids,
                "l2l_att_mask":iotensor.l2l_att_mask,
                "l2g_att_mask":iotensor.l2g_att_mask,
                "g2l_att_mask":iotensor.g2l_att_mask,
                "g2g_att_mask":iotensor.g2g_att_mask,
                "l2l_relative_att_ids":iotensor.l2l_relative_att_ids,
                "l2g_relative_att_ids":iotensor.l2g_relative_att_ids,
                "g2l_relative_att_ids":iotensor.g2l_relative_att_ids,
                "g2g_relative_att_ids":iotensor.g2g_relative_att_ids
        }
        labels={
                "label_id":iotensor.label_id,
                "global_label_id":iotensor.global_label_id
        }
        if FULLCONSTIT and not WSD:
            modifier="constitfull"
        elif FULLCONSTIT and WSD:
            modifier="truefull"
        elif not FULLCONSTIT and WSD:
            modifier="anno"
        else:
            print("Use constitfull, truefull, or anno")
            exit()
        with open(TOKENIZER+modifier+outpath+str(i)+".pickle",'wb') as f:
            pickle.dump((features,labels),f);


#writeexamples(jsontrainobject,"../DocREDtrain/train_annotated","examples/tf_example",annotate=True)
#writeexamples(jsondevobject,"../DocREDdev/dev","devexamples/tf_example",annotate=True)
writeexamples(jsonREtrainobject,"../REDocRED/train_revised","REexamples/tf_example",annotate=True)
writeexamples(jsonREdevobject,"../REDocRED/dev_revised","REdevexamples/tf_example",annotate=True)
#This code BADLY needs refactoring. I'll do it if I have time.
