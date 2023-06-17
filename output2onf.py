import os
import time
import json
import sys
import pickle
#Take in stanford .txt.out, CoNSeC .wsd.out, .psense.plabel.json, and .list
#return .onf

import nltk
from nltk.corpus import wordnet as wn

def insidebracket(s):#remove everything outside the brackets from a string
    result="";
    add=False;
    for c in s:
        if c=="]":
            return result;
        if add==True:
            result+=c
        if c=="[":
            add=True;
    return result;

def striptonum(s):#strip non numerals out of a string and convert the remaining numerals to int
    result="";
    for c in s:
        if c>='0' and c<='9':
            result+=c;
    return int(result);

class OntonotesAnnotation:
    def __init__(self,sentences,toktext,constit,coref,srl,wsd,entities):
        self.sentences=sentences;#list of raw sentences
        self.toktext=toktext;#tokenized and sentence split and lemmatized text(2d array of word/lemma tuples) (in order)
        self.constit=constit;#list of treebank trees (string form) (in order)
        self.coref=coref;#2d array of coref chains, each entry is tuple of (sentence#, starttok#,endtok#) (NOT in order)
        self.srl=srl;#propbank propositions (same structure as JSON srl)
        self.wsd=wsd;#senses for words (sentence#,word#,sensekey) (in order)
        self.entities=entities;#entity linking/categorization (sentencenum,starttoken,endtoken,type) (in order)

#read the stanford output, get tokenized text, constituent trees, and entity info, store in annotation
def readtxtout(filename,annotation):
    f=open(filename,'r',encoding="utf-8");
    lines=f.readlines();
    f.close();
    mode="None";#possible reading modes: None, Tokens, Constit
    curtree="";#current constituent tree
    curtoktext=[];#current tokenized sentence
    curentityinfo=[];#current info on entities in the sentence, a list of entity types
    curcorefchain=[];#current chain of coreference
    sentencenum=-1;#index of current sentence
    lines.append("END");#end of file symbol
    for line in lines:
        if "Constituency parse:" in line:
            mode="Constit";
        elif "Tokens:" in line:
            mode="Tokens";
        elif ("Sentence #" in line and " tokens):" in line) or line=="END":
            mode="None";
            #make sense of the entities in the sentence and add them to annotation
            curentitytype="O";
            curentitystart=-1;
            for i in range(len(curentityinfo)):
                if curentityinfo[i]=="O" and curentitytype!="O":#add our new entity to list
                    annotation.entities.append((sentencenum,curentitystart,i-1,curentitytype));
                elif curentitytype=="O":#new entity is starting
                    curentitystart=i;
                    curentitytype=curentityinfo[i];
                elif curentitytype!=curentityinfo[i]:#entity is different from previous, add it to list, also new entity is starting
                    annotation.entities.append((sentencenum,curentitystart,i-1,curentitytype));
                    curentitystart=i;
                curentitytype=curentityinfo[i];
            curentityinfo=[];
            #add sentence to list of sentences
            if line!="END":
                sentencenum+=1;
                annotation.sentences.append(lines[lines.index(line)+1])
        elif "Coreference set:" in line:
            mode="Coref";
            if curcorefchain!=[]:
                annotation.coref.append(curcorefchain);
                curcorefchain=[];
        elif line=="\n":
            if mode=="Constit":
                annotation.constit.append(curtree);
                curtree="";
            elif mode=="Tokens":
                annotation.toktext.append(curtoktext);
                curtoktext=[];
            mode="None";
        elif mode=="Constit":
            curtree+=line;
        elif mode=="Tokens" and line[0]=='[':
            lemma="";#lemma
            text="";#raw word
            entity="";#entity info(tag or O)
            line=insidebracket(line);#strip outside of brackets from the line
            lpairs=line.split(" ");
            for i in range(len(lpairs)):
                lpairs[i]=lpairs[i].split("=");
                if lpairs[i][0]=="Lemma":
                    lemma=lpairs[i][1];
                elif lpairs[i][0]=="Text":
                    text=lpairs[i][1];
                elif lpairs[i][0]=="NamedEntityTag":
                    entity=lpairs[i][1];
            curtoktext.append((text,lemma));
            curentityinfo.append(entity);
        elif mode=="Coref":#take in coref data, must subtract 1 from all numbers to index by zero, -2 from the end so it's inclusive
            line=line.split(" ");
            if len(line)<2 or line[1]!="->":#bad format, ignore
                print("bad coref format");
                continue;
            if curcorefchain==[]:#add the base to it
                nums=line[2].split(",");
                curcorefchain.append((striptonum(nums[0])-1,striptonum(nums[2])-1,striptonum(nums[3])-2));
            #now add the new coref to the chain
            nums=line[0].split(",");
            curcorefchain.append((striptonum(nums[0])-1,striptonum(nums[2])-1,striptonum(nums[3])-2));


#read the wsd folder, return a 2d array of wordnet codes for each term disambiguated("" for those not disambiguated)
#stores it in annotation
def readwsdout(filename,annotation):
    f=open(filename,'r',encoding="utf-8");
    lines=f.readlines();
    f.close();
    for line in lines:
        if line!="":#read in a line, convert it to a wsd triple (sentence#,word#,sensekey), add this to wsd
            line=line.split(" ");
            line[0]=line[0].split(".");
            annotation.wsd.append((striptonum(line[0][1]),striptonum(line[0][2]),line[1]))

def readsrl(jsonfilename,listfilename,annotation):
    f=open(jsonfilename,'r',encoding="utf-8");
    s=f.read();
    f.close();
    jsoninfo=json.loads(s);
    #align frameset_ids with predictions
    for sentence in jsoninfo:
        sentence["frameset_ids"]=[];
        for predicate in sentence["plemma_ids"]:
            sentence["frameset_ids"].append(predicate.split(".")[1]);
        #remove junk from argument list
        for i in range(len(sentence["arguments"])):
            sentence["arguments"][i]=[[sentence["predicates"][i],sentence["predicates"][i],"V"]]
    #align arguments with predictions in listfile
    f=open(listfilename,'r',encoding="utf-8");
    s=f.read();
    f.close();
    #change () to [] so we can treat this as json
    t="";
    for c in s:
        if c=="(":
            t+="[";
        elif c==")":
            t+="]";
        elif c=="'":
            t+='"';
        else:
            t+=c;
    listinfo=json.loads(t);
    #now read the arguments
    for arg in listinfo:
        sentence=jsoninfo[arg[0]];#sentence is the first argument
        if arg[1] in sentence["predicates"]:#this should always be true, arguments can only be on predicates
            sentence["arguments"][sentence["predicates"].index(arg[1])].append([arg[2],arg[3],arg[4]])
        else:
            print("Predicate index error")
    annotation.srl=jsoninfo;#put the data in the annotation


def synsetstrip(s):#take in a synset string, strip off Synset('')
    result="";
    active=False;#currently adding chars or not?
    for c in s:
        if c=='(':
            active=True;
        elif c==')':
            active=False;
        elif active and c!="'":
            result+=c;
    return result;

#Convert ontonotes annotation object to onf string and store it in outfile
def annotation2onf(annotation,outfile):
    outstring=""#string we're outputting to the file
    separator="------------------------------------------------------------------------------------------------------------------------\n\n"
    for i in range(len(annotation.sentences)):#print the sentence, the tree section and leaves section for the sentence
        outstring+=separator+"Plain sentence:\n---------------\n"
        outstring+=annotation.sentences[i]+"\n\n"
        outstring+="Treebanked sentence:\n--------------------\n"
        for tok in annotation.toktext[i]:
            outstring+=tok[0]+" ";
        outstring+="\n\n";
        outstring+="Tree:\n-----\n"
        outstring+=annotation.constit[i]+"\n\n";
        #now is the leaves section, the more complicated part.
        sentence=annotation.toktext[i];
        outstring+="Leaves:\n-------\n";
        for j in range(len(sentence)):
            outstring+="\t"+str(j)+"\t"+sentence[j][0]+"\n";
            #Search for coref, name, prop, sense
            for k in range(len(annotation.coref)):#coref search
                for coref in annotation.coref[k]:
                    if coref[0]==i and coref[1]==j:
                        outstring+="\t\t\tcoref:\tIDENT\t\t\t"+str(k)+"\t"+str(coref[1])+"-"+str(coref[2])+"\t";
                        for l in range(coref[1],coref[2]+1):
                            outstring+=sentence[l][0]+" ";
                        outstring+="\n";
            for name in annotation.entities:#name search
                if name[0]==i and name[1]==j:
                    outstring+="\t\t\tname:\t"+name[3]+"\t\t\t\t"+str(name[1])+"-"+str(name[2])+"\t";
                    for k in range(name[1],name[2]+1):
                        outstring+=sentence[k][0]+" ";
                    outstring+="\n";
            for sense in annotation.wsd:#sense search
                if sense[0]==i and sense[1]==j:
                    outstring+="\t\t\tsense:\t"+synsetstrip(str(wn.synset_from_sense_key(sense[2])))+"\n";#turn this sensekey into an ontonotes-style sense
            if annotation.srl==[]:#case with no propositions, don't do the check
                pass;
            elif j in annotation.srl[i]["predicates"]:#proposition check
                pindex=annotation.srl[i]["predicates"].index(j);
                outstring+="\t\t\tprop:\t"+annotation.srl[i]["plemma_ids"][pindex]+"\n";#add proposition sense
                for arg in annotation.srl[i]["arguments"][pindex]:#TODO: convert the spans into treenodes
                    outstring+="\t\t\t"+arg[2]+"\t\t"+str(arg[0])+"-"+str(arg[1])+"\t";
                    for k in range(arg[0],arg[1]+1):
                        if k>=len(sentence):#FIX: The indexing from srl is still really strange...
                            print(i,j,arg,sentence);
                        else:
                            outstring+=sentence[k][0]+" ";
                    outstring+="\n";
        outstring+="\n\n";
    #add coreference section at end
    outstring+="========================================================================================================================\n"
    outstring+="Coreference chains for section 0:\n"
    outstring+="---------------------------------\n\n"
    for i in range(len(annotation.coref)):
        outstring+="\t"+"Chain "+str(i)+" (IDENT)\n"
        for coref in annotation.coref[i]:
            outstring+="\t\t\t\t"+str(coref[0])+"."+str(coref[1])+"-"+str(coref[2])+"\t";
            for j in range(coref[1],coref[2]+1):
                outstring+=annotation.toktext[coref[0]][j][0]+" ";
            outstring+="\n";
    #write to file
    print(outstring,file=open(outfile,'w',encoding="utf-8"));

if(len(sys.argv)>=6):#command, 4 input filenames, 1 output filename
    annotation=OntonotesAnnotation([],[],[],[],[],[],[])
    readtxtout(sys.argv[1],annotation)
    readwsdout(sys.argv[2],annotation)
    readsrl(sys.argv[3],sys.argv[4],annotation)
    annotation2onf(annotation,sys.argv[5])
elif len(sys.argv)>1 and sys.argv[1][len(sys.argv[1])-1]=='/':#input directory for continuous action
    name=""#path to data without extension
    annotation=OntonotesAnnotation([],[],[],[],[],[],[])
    flist=os.listdir(sys.argv[1]);

    for path in flist:
        if len(path)>=8 and path[len(path)-8:len(path)]==".txt.out":
            name=path[0:len(path)-8];
        elif len(path)>=8 and path[len(path)-8:len(path)]==".wsd.out":
            name=path[0:len(path)-8];
        elif len(path)>=19 and path[len(path)-19:len(path)]==".psense.plabel.json":
            name=path[0:len(path)-19];
        elif len(path)>=5 and path[len(path)-5:len(path)]==".list":
            name=path[0:len(path)-5];
        if name!="" and name+".txt.out" in flist and name+".wsd.out" in flist and name+".list" in flist:
            txtpath=sys.argv[1]+name+".txt.out";
            wsdpath=sys.argv[1]+name+".wsd.out";
            jsonpath=sys.argv[1]+name+".psense.plabel.json";
            listpath=sys.argv[1]+name+".list";
            readtxtout(txtpath,annotation);
            readwsdout(wsdpath,annotation);
            if(name+".psense.plabel.json" in flist):#if we have a listpath but no jsonpath, that means there were no frames and we can skip this part.
                readsrl(jsonpath,listpath,annotation);
            annotation2onf(annotation,sys.argv[1]+name+".onf");
            #Also dump to json file for better computer reading
            jsonobj={}
            jsonobj['sentences']=annotation.sentences
            jsonobj['toktext']=annotation.toktext
            jsonobj['constit']=annotation.constit
            jsonobj['coref']=annotation.coref
            jsonobj['srl']=annotation.srl
            jsonobj['wsd']=annotation.wsd
            jsonobj['entities']=annotation.entities
            with open(sys.argv[1]+name+".final.json",'w') as f:
                json.dump(jsonobj,f);
            annotation=OntonotesAnnotation([],[],[],[],[],[],[]);
elif len(sys.argv)==1:#Use this to convert the pickle into json, which is a better format for this application
    for i in range(998):
        with open("DocREDdev/dev"+str(i)+".pickle",'rb') as f:
            obj=pickle.load(f)
            jsonobj={}
            jsonobj['sentences']=obj.sentences
            jsonobj['toktext']=obj.toktext
            jsonobj['constit']=obj.constit
            jsonobj['coref']=obj.coref
            jsonobj['srl']=obj.srl
            jsonobj['wsd']=obj.wsd
            jsonobj['entities']=obj.entities
        with open("DocREDdev/dev"+str(i)+".final.json",'w') as f:
            json.dump(jsonobj,f)
        print(i)
else:
    print("Not enough arguments to output2onf")
