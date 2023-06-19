import argparse

#used for automatically finding definitions from wordnet
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import hydra
import pytorch_lightning as pl
from typing import Iterator, Tuple, List, Optional

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

#ADDED!
import sys
sys.path.append("/nfs/hpc/share/saueran/personal/Thesis/consec-main")
#ADDED!

from src.consec_dataset import ConsecDataset, ConsecSample, ConsecDefinition
from src.disambiguation_corpora import DisambiguationInstance
from src.pl_modules import ConsecPLModule
from src.consec_tokenizer import DeBERTaTokenizer, ConsecTokenizer


def predict(
    module: pl.LightningModule,
    tokenizer: ConsecTokenizer,
    samples: Iterator[ConsecSample],
    text_encoding_strategy: str,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Iterator[Tuple[ConsecSample, List[float]]]:

    # todo only works on single gpu
    device = next(module.parameters()).device

    # todo hardcoded dataset
    dataset = ConsecDataset.from_samples(
        samples,
        tokenizer=tokenizer,
        use_definition_start=True,
        text_encoding_strategy=text_encoding_strategy,
        tokens_per_batch=token_batch_size,
        max_batch_size=128,
        section_size=2_000,
        prebatch=True,
        shuffle=False,
        max_length=tokenizer.model_max_length,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # predict

    iterator = dataloader
    progress_bar = tqdm() if progress_bar else None

    for batch in iterator:

        batch_samples = batch["original_sample"]
        batch_definitions_positions = batch["definitions_positions"]

        with autocast(enabled=True):
            with torch.no_grad():
                batch_out = module(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                batch_predictions = batch_out["pred_probs"]

        for sample, dp, probs in zip(batch_samples, batch_definitions_positions, batch_predictions):
            definition_probs = []
            for start in dp:
                definition_probs.append(probs[start].item())
            yield sample, definition_probs
            if progress_bar is not None:
                progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()


def interactive_main(
    model_checkpoint_path: str,
    device: int,
):
    def read_ld_pairs() -> List[Tuple[str, str, Optional[str]]]:
        pairs = []
        while True:
            line = input(" * ").strip()
            if line == "":
                break
            parts = line.split(" --- ")
            if len(parts) == 3:
                l, d, p = parts
                p = int(p)
            elif len(parts) == 2:
                l, d = parts
                p = None
            else:
                raise ValueError
            pairs.append((l, d, p))
        return pairs

    # load model
    # todo decouple BasecPLModule
    module = ConsecPLModule.load_from_checkpoint(model_checkpoint_path)
    module.to(torch.device(device if device != -1 else "cpu"))
    module.freeze()
    module.sense_extractor.evaluation_mode = True

    # load tokenizer
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    while True:

        # read marked text
        text = input("Enter space-separated text: ").strip()
        tokens = text.split(" ")
        target_position = int(input("Target position: ").strip())

        # read candidates definitions
        print('Enter candidate lemma-def pairs. " --- " separated. Enter to stop')
        candidate_definitions = read_ld_pairs()
        candidate_definitions = [ConsecDefinition(d, l) for l, d, _ in candidate_definitions]

        # read context definitions
        print(
            'Enter context lemma-def-position tuples. " --- " separated. Position should be token position in space-separated input. Enter to stop'
        )
        context_definitions = read_ld_pairs()
        context_definitions = [(ConsecDefinition(d, l), p) for l, d, p in context_definitions]

        # predict
        _, probs = next(
            predict(
                module,
                tokenizer,
                [
                    ConsecSample(
                        sample_id="interactive-d0",
                        position=target_position,
                        disambiguation_context=[
                            DisambiguationInstance("d0", "s0", "i0", t, None, None, None) for t in tokens
                        ],
                        candidate_definitions=candidate_definitions,
                        gold_definitions=None,
                        context_definitions=context_definitions,
                        in_context_sample_id2position={'interactive-d0': target_position},
                        disambiguation_instance=None,
                        kwargs={},
                    )
                ],
                text_encoding_strategy="simple-with-linker",  # todo hardcoded core param
            )
        )

        idxs = torch.tensor(probs).argsort(descending=True)
        print(f"\t# predictions")
        for idx in idxs:
            idx = idx.item()
            print(
                f"\t\t * {probs[idx]:.4f} \t {candidate_definitions[idx].linker} \t {candidate_definitions[idx].text} "
            )


def file_main(
    model_checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: int,
    token_batch_size: int,
):
    #raise NotImplementedError
    def get_all_ld_pairs_tokens(input_path) -> Tuple[List[List[Tuple[str,str]]],List[str]]:#extract the lemmas and candidate definitions from the Stanford-annotated input file
        #normalize the word by lowercasing and stripping off punctuation
        input_file=open(input_path,'r');
        lines=input_file.readlines();
        input_file.close()
        readtoks=False;#are we currently reading tokens?
        result=[];
        tokens=[];
        for line in lines:
            if "Tokens:" in line:
                readtoks=True;
            elif line[0]=='[' and readtoks==True:
                lemma="";#lemma for token
                pos="";#part of speech
                lpairs=line.split(' ');
                for i in range(len(lpairs)):
                    lpairs[i]=lpairs[i].split('=');
                    if lpairs[i][0]=="Lemma":
                        lemma=lpairs[i][1];
                    if lpairs[i][0]=="PartOfSpeech":
                        pos=lpairs[i][1];
                    if lpairs[i][0]=="[Text":
                        tokens.append(lpairs[i][1]);#add token to list
                #now find the list of possible meanings for the lemma based on pos
                #change from treebank codes to wordnet codes
                if pos=="NN" or pos=="NNS":
                    pos=wn.NOUN;#noun codes(not proper nouns, we use entity linking for that)
                elif pos=="JJ" or pos=="JJR" or pos=="JJS":
                    pos=wn.ADJ;#adjective codes
                elif pos=="RB" or pos=="RBR" or pos=="RBS":
                    pos=wn.ADV;#adverb codes
                elif pos=="VB" or pos=="VBD" or pos=="VBG" or pos=="VBN" or pos=="VBP" or pos=="VBZ":
                    pos=wn.VERB;#verb codes
                else:
                    result.append([])#no WSD for this if it's not a noun, adjective, adverb or verb
                    continue;
                synsets=wn.synsets(lemma,pos=pos)
                definitions=[];
                for synset in synsets:
                    definitions.append((lemma,synset.definition()));
                result.append(definitions);#add all the possible definitions of this lemma for this pos to the result
            else:
                readtoks=False;
        return (result,tokens);

    #load model
    module=ConsecPLModule.load_from_checkpoint(model_checkpoint_path)
    module.to(torch.device(device if device != -1 else "cpu"))
    module.freeze();
    module.sense_extractor.evaluation_mode=True

    #load tokenizer
    tokenizer=hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    ld_pairs_tokens=get_all_ld_pairs_tokens(input_path);
    tokambigs=ld_pairs_tokens[0]#list of possible interpretations from WordNet for each token
    tokens=ld_pairs_tokens[1]#list of all tokens
    #now go through, disambiguate the ones with fewer interpretations first
    #then input those to the next disambiguations
    disambigs=[];#list of already made disambiguation
    numinterp=1;#number of interpretations for the words we're disambiguating now
    while True:
        done=True;#determine when we've disambiguated all the words that can be
        samples=[]#list of consec samples with this particular number of interpretations
        positions=[]#positions of all the samples
        for i in range(len(tokambigs)):#make consec samples for all the tokens with numinterp interpretations
            if len(tokambigs[i])==numinterp:
                #code copied and modified a bit from interactive
                candidate_definitions=[ConsecDefinition(d,l) for l,d in tokambigs[i]]
                context_definitions=[(ConsecDefinition(d,l),p) for l,d,p in disambigs]
                N=10; #context window width
                position=-1; #position of word within token_context
                token_context=[];#list of tokens surrounding the current token, N in each direction
                for j in range(-N,N+1):
                    if i+j>=0 and i+j<len(tokens):
                        if j==0:
                            position=len(token_context)#this is the position of the target token
                        token_context.append(tokens[i+j])
                tmp=[];#remove definitions from outside the window
                for definition in context_definitions:
                    if definition[1]>=i-N and definition[1]<=i+N:
                        if i>N:
                            definition=(definition[0],definition[1]-i+N)#position needs to line up to the context window
                        tmp.append(definition);
                context_definitions=tmp;
                samples.append(ConsecSample(
                    sample_id="interactive-d0",
                    position=position,
                    disambiguation_context=[DisambiguationInstance("d0","s0","i0",t,None,None,None) for t in token_context],
                    candidate_definitions=candidate_definitions,
                    gold_definitions=None,
                    context_definitions=context_definitions,
                    in_context_sample_id2position={'interactive-d0': position},
                    disambiguation_instance=None,
                    kwargs={},
                    ))
            if len(tokambigs[i])>numinterp:
                done=False;
        #predict all these samples at once, more efficient
        samproblist=[];
        numsamples=len(samples)
        samples=iter(samples)
        predictiter=predict(module,tokenizer,samples,text_encoding_strategy="simple-with-linker",)#iter object returned from predicting all at once
        for i in range(numsamples):
            sample,probs=next(predictiter)
            samproblist.append((sample,probs))
        for sample,probs in samproblist:
            idxs=torch.tensor(probs).argsort(descending=True)
            maxprob=0;
            maxtup=(0,0,0);#context tuple for the maximum-probability
            for idx in idxs:
                idx=idx.item()
                if probs[idx]>maxprob:
                        #change the context tuple
                    maxprob=probs[idx]
                    maxtup=(sample.candidate_definitions[idx].linker,sample.candidate_definitions[idx].text,-1)#position isn't currently working well
            print(maxtup);
            disambigs.append(maxtup);
        numinterp+=1;
        if done:#no more larger disambiguations
            break;
    outstring="";#prettify the info and print it to a file
    for i in range(len(tokens)):
        outstring+=tokens[i]
        tokdef="";
        for disambig in disambigs:
            if disambig[2]==i:
                outstring+='(';
                outstring+=disambig[1];
                outstring+=')';
                break;
        if tokens[i]=='.' or tokens[i]=='?' or tokens[i]=='!':
            outstring+='\n\n';
        else:
            outstring+=' ';
    print(outstring,file=open(output_path,'w'))
def main():
    args = parse_args()
    if args.t:
        interactive_main(
            args.model_checkpoint,
            device=args.device,
        )
    else:
        file_main(
            args.model_checkpoint,
            args.f,
            args.o,
            device=args.device,
            token_batch_size=args.token_batch_size,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--device", type=int, default=-1, help="Device")
    # interactive params
    parser.add_argument("-t", action="store_true", help="Interactive mode")
    # generation params
    parser.add_argument("-f", type=str, default=None, help="Input file")
    parser.add_argument("-o", type=str, default=None, help="Output file")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
