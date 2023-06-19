import sys
import os

endstrings=["3053/3053","611/611","306/306"]
maximum=True#Should we only print the maximum among epochs?
DO_PRINT=False

#take in string, determine max f1, max ign f1, threshold, train loss, val loss
#do_print is boolean which determines if we actually print
def print_info_from_string(s,do_print):
    global epochmaxf1, epochf1
    global epochmaxignf1, epochignf1
    global cur_epoch
    toks=[""]
    for c in s:
        if c in " \t\n" and toks[-1]!="":
            toks.append("")
        elif not c in " \t\n":
            toks[-1]+=c
    if toks[-1]=="":
        toks.pop()
    results={}
    for i in range(len(toks)):
        if toks[i][-1]==":" and i<len(toks)-1:
            try:
                results[toks[i]]=float(toks[i+1])
            except:
                pass
    keys=results.keys()
    if "loss:" in keys and do_print:
        print("Train loss: "+str(results["loss:"]))
    if "val_loss:" in keys and do_print:
        print("Validation loss: "+str(results["val_loss:"]))
    if "val_AFLprecision:" in results.keys() and "val_AFLrecall:" in results.keys():
        p=results["val_AFLprecision:"]
        r=results["val_AFLrecall:"]
        if do_print:
            print("AFLprecision: "+str(p))
            print("AFLrecall: "+str(r))
            print("AFL F1: "+str((2*p*r)/(p+r)))
        if (2*p*r)/(p+r)>epochmaxf1:
            epochmaxf1=(2*p*r)/(p+r)
            epochf1=cur_epoch
    if "val_ignAFLprecision:" in results.keys() and "val_ignAFLrecall:" in results.keys():
        p=results["val_ignAFLprecision:"]
        r=results["val_ignAFLrecall:"]
        if do_print:
            print("ignAFLprecision: "+str(p))
            print("ignAFLrecall: "+str(r))
            print("AFL IgnF1: "+str((2*p*r)/(p+r+.0001)))
        if (2*p*r)/(p+r)>epochmaxignf1:
            epochmaxignf1=(2*p*r)/(p+r)
            epochignf1=cur_epoch
        if do_print:
            print('\n')
        return
    maxf1=0.0
    threshold=-100
    for i in range(21):
        pkey="val_"+str(i*0.5-5.0)+"precision:"
        rkey="val_"+str(i*0.5-5.0)+"recall:"
        if pkey in keys and rkey in keys:
            f1=2.0*(results[pkey]*results[rkey])/(results[pkey]+results[rkey]+.0001)
            if f1>maxf1:
                maxf1=f1
                threshold=i*0.5-5.0
    if do_print:
        print("Max F1: "+str(maxf1))
        print("Threshold: "+str(threshold))
    if maxf1>epochmaxf1:
        epochmaxf1=maxf1
    maxf1=0.0
    threshold=-100
    for i in range(21):
        pkey="val_ign"+str(i*0.5-5.0)+"precision:"
        rkey="val_ign"+str(i*0.5-5.0)+"recall:"
        if pkey in keys and rkey in keys:
            f1=2.0*(results[pkey]*results[rkey])/(results[pkey]+results[rkey]+.0001)
            if f1>maxf1:
                maxf1=f1
                threshold=i*0.5-5.0
    if do_print:
        print("Max Ign F1: "+str(maxf1))
        print("Ign Threshold: "+str(threshold))
    if maxf1>epochmaxignf1:
        epochmaxignf1=maxf1
    if do_print:
        print('\n')

BASE_FILENAME="12L1B15E2e-05LR1.0gAFL5FinPreAdamW"
CSV_FILE="statistics.csv"
g=open(CSV_FILE,'w')
h=open("ign_"+CSV_FILE,'w')
for i in range(1,31):
    for j in ["","ANNO","ANNOFULL"]:
        filename="BERTstatistics/"+j+BASE_FILENAME+str(i)+".txt"
        epochmaxf1=0
        epochf1=0
        epochmaxignf1=0
        epochignf1=0
        cur_epoch=0
        try:
            with open(filename) as f:
                lines=f.readlines()
                for line in lines:
                    if "Epoch" in line:
                        cur_epoch+=1
                        if DO_PRINT:
                            print(line)
                    for endstring in endstrings:
                        if endstring in line and "val" in line:
                            index=line.rindex(endstring)
                            try:
                                print_info_from_string(line[index:],DO_PRINT)
                            except Exception as e:
                                print(e)
        except Exception as e:
            print(e)
        if j=="ANNOFULL":
            print(epochmaxf1,file=g)
            print(epochmaxignf1,file=h)
        else:
            print(epochmaxf1,file=g,end=',')
            print(epochmaxignf1,file=h,end=',')
#
