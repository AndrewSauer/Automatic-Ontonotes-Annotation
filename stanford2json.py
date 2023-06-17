import sys
import os
import time
#take in a stanford-formatted input file, output xml of the document with bare text, lemmas, pos, tokenization, sentence splitting

#input list, output json list string
def list2json(l):
    result="["
    for i in range(len(l)):
        if i>0:
            result+=",";
        if type(l[i])==str:
            result+='"';
            for c in l[i]:
                if c=='"':
                    result+="'";
                else:
                    result+=c;
            result+='"';
        elif type(l[i])==list:
            result+=list2json(l[i]);
        else:
            result+=str(l[i]);
    result+="]"
    return result;

def escape(s):#replace all instances of \ with \\, raw text doesn't use escapes
    result="";
    for c in s:
        if c=='\\':
            result+='\\\\';
        else:
            result+=c;
    return result;

def stanfordtojson(inputpath,outputpath):
    outstring="[";
    inputfile=open(inputpath,'r');
    lines=inputfile.readlines();
    inputfile.close();
    sentencenum=0;
    wordnum=0;
    readtoks=False;
    predicates=[];#list of srl verb positions
    lemmas=[];#list of lemmatized srl verbs
    sentence=[];#list of tokens
    for line in lines:
        if "Tokens:" in line:
            readtoks=True;
        elif line[0]=='[' and readtoks==True:
            lemma="";
            pos="";
            lpairs=line.split(' ');
            for i in range(len(lpairs)):
                #special case for the = sign: if == in lpairs the lemma/text is'='
                if '==' in lpairs[i]:
                    lemma='=';
                    pos='SYM';
                    sentence.append('=')
                    break;
                lpairs[i]=lpairs[i].split('=');
                if lpairs[i][0]=="Lemma":
                    lemma=escape(lpairs[i][1]);
                if lpairs[i][0]=="PartOfSpeech":
                    pos=lpairs[i][1];
                if lpairs[i][0]=="[Text":
                    sentence.append(escape(lpairs[i][1]));
            if pos=="VB" or pos=="VBD" or pos=="VBG" or pos=="VBN" or pos=="VBP" or pos=="VBZ":
                predicates.append(wordnum);
                lemmas.append(lemma);
            wordnum+=1;
        else:
            if(readtoks==True):#we're done with a sentence, encode it and add it to the outstring
                frameset_ids=[];#put in fake golds for the evals to "compare against"
                arguments=[];
                for i in range(len(predicates)):
                    frameset_ids.append("01");
                    arguments.append([[0,0,"ARG0"],[predicates[i],predicates[i],"V"]]);

                #add our lists to the json outstring
                if sentencenum!=0:
                    outstring+=",";
                outstring+="\n\t{\n";
                outstring+="\t\t\"arguments\": "+list2json(arguments)+",\n";
                outstring+="\t\t\"frameset_ids\": "+list2json(frameset_ids)+",\n";
                outstring+="\t\t\"lemmas\": "+list2json(lemmas)+",\n";
                outstring+="\t\t\"predicates\": "+list2json(predicates)+",\n";
                outstring+="\t\t\"sentence\": "+list2json(sentence)+"\n";
                outstring+="\t}";
                sentencenum+=1;
                wordnum=0;
            predicates=[];
            lemmas=[];
            sentence=[];
            readtoks=False;
    outstring+="\n]"
    outputfile=open(outputpath,'w');
    outputfile.write(outstring);
    outputfile.close();

if(len(sys.argv))>=3:
    print(sys.argv[1],sys.argv[2])
    stanfordtojson(sys.argv[1],sys.argv[2])
elif sys.argv[1][len(sys.argv[1])-1]=='/':
    #operate on every .txt.out currently in directory
    flist=os.listdir(sys.argv[1]);
    for path in flist:
        if len(path)>=8 and path[len(path)-8:len(path)]==".txt.out" and not path[0:len(path)-8]+".json" in flist and not path[0:len(path)-8]+".psense.json" in flist and not path[0:len(path)-8]+".psense.plabel.json" in flist and not path[0:len(path)-8]+".list" in flist and not path[0:len(path)-8]+".onf" in flist:
            stanfordtojson(sys.argv[1]+path,sys.argv[1]+path[0:len(path)-8]+".json");
else:
    print("Error: not enough arguments to stanfordtojson!")
