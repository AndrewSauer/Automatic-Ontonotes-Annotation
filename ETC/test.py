import pickle

CONSTIT_TYPES=['ROOT', 'S', 'NP', 'NNP', 'NNPS', 'PRN', ',', 'VP', 'VBD', 'PP', 'IN', '-LRB-', 'RB', 'JJ', 'CC', '-RRB-', 'DT', 'NML', 'HYPH', 'NN', 'VBN', '.', 'PRP', 'ADJP', 'NNS', 'VBG', 'CD', 'PRP$', 'ADVP', 'TO', 'VB', 'QP', 'JJR', 'POS', 'VBZ', 'SBAR', 'VBP', 'EX', 'RBR', 'NP-TMP', 'WHNP', 'WDT', '``', "''", 'JJS', 'WP', 'SINV', 'FRAG', 'WHADVP', 'WRB', 'WHPP', 'PRT', 'RP', 'UCP', 'WP$', ':', 'FW', 'RRC', 'SYM', '$', 'CONJP', 'AFX', 'SBARQ', 'SQ', 'RBS', 'NAC', 'NFP', 'MD', 'PDT', 'UH', 'INTJ', 'ADD', 'NX', 'GW', 'X', 'LST', 'LS', 'WHADJP']
SRL_ARGUMENTS=['V','A0','A1','A2','A3','A4','A5','LOC','ADV','TMP','MNR','PRD','CAU','DIS','GOL','COM','EXT','PRP','DIR','NEG','PNC','MOD','REC','ADJ','LVB']
WSD_TYPES=pickle.load(open("senses.pickle",'rb'))
SRL_TYPES=pickle.load(open("frames.pickle",'rb'))
print(len(CONSTIT_TYPES))
print(len(SRL_ARGUMENTS))
print(len(WSD_TYPES))
print(len(SRL_TYPES))
