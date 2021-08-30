import nltk
import nltk.corpus.treebank as tb
import six
import numpy as np

#10% sample of the Penn Treebank
ruleset = list(set(rule for tree in tb.parsed_sents() for rule in tree.productions()))

start_index = ruleset[0].lhs()

all_lhs = [a.lhs().symbol() for a in ruleset]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(ruleset)

rhs_map = [None] * D
count = 0
for a in ruleset:
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b, six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list), D))
count = 0
# all_lhs.append(0)
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:, i] == 1)[0][0])
ind_of_ind = np.array(index_array)
