import os
import sys

#DON'T use until testing on small dataset
#exit()
if sys.argv[1][len(sys.argv[1])-1]=='/':
    name="";
    flist=os.listdir(sys.argv[1]);
    txtfilelist=""
    for path in flist:
        if len(path)>=4 and path[len(path)-4:len(path)]==".onf":
            name=path[0:len(path)-4]
            if name+".txt" in flist:
                os.rename(sys.argv[1]+name+".txt",sys.argv[1]+"txtfiles/"+name+".txt")
            if name+".txt.out" in flist:
                os.remove(sys.argv[1]+name+".txt.out")
            if name+".wsd.out" in flist:
                os.remove(sys.argv[1]+name+".wsd.out")
            if name+".data.xml" in flist:
                os.remove(sys.argv[1]+name+".data.xml")
            if name+".json" in flist:
                os.remove(sys.argv[1]+name+".json")
            if name+".psense.json" in flist:
                os.remove(sys.argv[1]+name+".psense.json")
            if name+".psense.plabel.json" in flist:
                os.remove(sys.argv[1]+name+".psense.plabel.json")
            if name+".list" in flist:
                os.remove(sys.argv[1]+name+".list")
