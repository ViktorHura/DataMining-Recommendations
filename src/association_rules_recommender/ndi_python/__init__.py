from itertools import chain, combinations, filterfalse
import subprocess
import math
import os
import pathlib


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def association_rules(transactions, min_support, min_confidence):
    min_support_abs = math.floor(min_support*len(transactions))

    path = pathlib.Path(__file__).parent.resolve()


    with open(os.path.join(path, 'in.dat'), 'w') as f:
        for t in transactions:
            line = ""
            for i in t:
                line += i + " "
            f.write(line + "\n")

    pout = subprocess.run([os.path.join(path, 'build', 'ndi'), os.path.join(path, 'in.dat'), str(min_support_abs), "10",os.path.join(path, 'out.txt')])

    frequent_itemsets = []
    supports = []
    with open(os.path.join(path, 'out.txt'), 'r') as f:
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            itemst = []
            buffer = ""
            support = None
            for c in line:
                if c == '(':
                    support = ""
                elif c ==')':
                    support = int(support)
                    break
                elif c == ' ':
                    itemst.append(buffer)
                    buffer = ""
                elif support is not None:
                    support += c
                else:
                    buffer += c

            frequent_itemsets.append(frozenset(itemst))
            supports.append(support)

    print("generating rules")

    rules = []
    for i, itemset in enumerate(frequent_itemsets):
        p = (i+1) / len(frequent_itemsets)
        if i % math.floor(len(frequent_itemsets)/100) == 0:
            print('\r' + str(math.floor(p*100)), end='')

        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            if consequent == frozenset():
                continue
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)])
            support_itemset = supports[i]
            confidence = support_itemset / support_antecedent
            if confidence >= min_confidence:
                rules.append([antecedent, consequent, support_itemset / len(transactions), confidence])
    print()
    return rules


def main():
    import pandas as pd
    transactions = pd.read_csv("../../../data/retail2-transactions.tsv", sep="\t", index_col=0)
    # yes the eval looks sketchy, it is to convert strings to sets
    transactions = [i for i in transactions['transaction'].apply(eval).tolist()]

    rules = association_rules(transactions, min_support=0.0025, min_confidence=0.6)

    print(f"Generated {len(rules)} rules")
    for antecedent, consequent, support, confidence in rules:
        print(f"{antecedent} => {consequent} (support={support:.2f}, confidence={confidence:.2f})")


if __name__ == '__main__':
    main()
