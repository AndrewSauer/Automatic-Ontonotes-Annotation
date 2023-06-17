#Preprocess all files by adding punctuation to every line which does not have it at the end.
#This is to prevent sentences from getting too long when punctuation is forgone.
#In general sentences cannot be too long or they will trip up the system. Very long sentences should be split up.
#TODO: truncate sentences with estimated length over 1000, and spit out a warning.
import sys
import os
def ispunc(c):
    return c in "~`!@#$%^&*()_-+=[]{}|\\;:\'\",.<>/?"
def iswhitespace(c):
    return c in " \t\n";
def addpunc(s):#add ' .' to the end of a string(before newline) if there is a letter not preceding '.', '!' or '?'.
    num_tokens=0;
    if s[len(s)-1]=='\n':
        s=s[0:len(s)-1];#strip off newline if it exists
    needspunc=False;
    for c in s:
        if (c>='A' and c<='Z') or (c>='a' and c<='z') or (c>='0' and c<='9'):
            needspunc=True;
            word=True;
        if c=='.' or c=='!' or c=='?':
            needspunc=False;
        if ispunc(c):
            num_tokens+=1;
        if iswhitespace(c):
            word=False;
            num_tokens+=1;
        if num_tokens>=1000:#break off sentence if there are too many tokens and warn
            print("Error: Sentence \""+s+"\" is too long! It has been concatenated!")
            break;
    if needspunc:
        return s+" .\n";
    else:
        return s+"\n";

if len(sys.argv)<2:
    print("Must specify directory to preprocess!")
else:
    flist=os.listdir(sys.argv[1]);
    for path in flist:
        if path[len(path)-4:]==".txt":
            with open(sys.argv[1]+path,'r') as f:#read through file, get punctuated string
                lines=f.readlines();
                newstring=""
                for line in lines:
                    newstring+=addpunc(line);
            with open(sys.argv[1]+path,'w') as f:
                f.write(newstring);#overwrite file with new punctuated string
