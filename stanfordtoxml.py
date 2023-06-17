import sys
import os
import time
#take in a stanford-formatted input file, output xml of the document with bare text, lemmas, pos, tokenization, sentence splitting

def xmlescape(s):#xml escape problematic chars out of a string
    result="";
    for c in s:
        if c=="\"":
            result+="&quot;"
        elif c=="<":
            result+="&lt;"
        elif c=="&":
            result+="&amp;"
        else:
            result+=c;
    return result;

def stanfordtoxml(inputpath,outputpath,lang,source):
    outstring="<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    outstring+="<corpus lang=\""+lang+"\" source=\""+source+"\">\n";
    outstring+="<text id=\"d0\">\n";
    inputfile=open(inputpath,'r');
    lines=inputfile.readlines();
    inputfile.close();
    sentencenum=0;
    wordnum=0;
    readtoks=False;
    for line in lines:
        if "Tokens:" in line:
            readtoks=True;
            outstring+="<sentence id=\"d0.s"+str(sentencenum)+"\">\n";
        elif line[0]=='[' and readtoks==True:
            lemma="";
            pos="";
            text="";
            lpairs=line.split(' ');
            for i in range(len(lpairs)):
                #special case for '=' character
                if '==' in lpairs[i]:
                    lemma='=';
                    pos='SYM';
                    text='=';
                lpairs[i]=lpairs[i].split('=');
                if lpairs[i][0]=="Lemma":
                    lemma=lpairs[i][1];
                if lpairs[i][0]=="PartOfSpeech":
                    pos=lpairs[i][1];
                if lpairs[i][0]=="[Text":
                    text=lpairs[i][1];
            if pos=="NN" or pos=="NNS":
                pos="NOUN";
            elif pos=="JJ" or pos=="JJR" or pos=="JJS":
                pos="ADJ";
            elif pos=="RB" or pos=="RBR" or pos=="RBS":
                pos="ADV";
            elif pos=="VB" or pos=="VBD" or pos=="VBG" or pos=="VBN" or pos=="VBP" or pos=="VBZ":
                pos="VERB";
            else:
                outstring+="<wf lemma=\""+xmlescape(lemma)+"\" pos=\""+xmlescape(pos)+"\">"+xmlescape(text)+"</wf>\n"
                wordnum+=1;
                continue;
            outstring+="<instance id=\"d0.s"+str(sentencenum)+".t"+str(wordnum)+"\" lemma=\""+xmlescape(lemma)+"\" pos=\""+xmlescape(pos)+"\">"+xmlescape(text)+"</instance>\n"
            wordnum+=1;
        else:
            if(readtoks==True):
                outstring+="</sentence>\n";
                sentencenum+=1;
                wordnum=0;
            readtoks=False;
    outstring+="</text>\n</corpus>"
    outputfile=open(outputpath,'w');
    outputfile.write(outstring);
    outputfile.close();

if(len(sys.argv))>=4:
    print(sys.argv[1],sys.argv[2],sys.argv[3])
    stanfordtoxml(sys.argv[1],sys.argv[2],"en",sys.argv[3])
elif sys.argv[1][len(sys.argv[1])-1]=='/':#inputted directories must end in / otherwise error is caused
    #operate on every .txt.out file currently there
    flist=os.listdir(sys.argv[1]);
    for path in flist:
        if len(path)>=8 and path[len(path)-8:len(path)]==".txt.out" and not path[0:len(path)-8]+".wsd.out" in flist and not path[0:len(path)-8]+".data.xml" in flist and not path[0:len(path)-8]+".onf" in flist:
            stanfordtoxml(sys.argv[1]+path,sys.argv[1]+path[0:len(path)-8]+".data.xml","en",path[0:len(path)-8]);
else:
    print("Error: not enough arguments to stanfordtoxml!")
