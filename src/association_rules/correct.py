from itertools import chain, combinations, filterfalse


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def join_set(itemsets, k):
    return set(
        [comb[0].union(comb[1]) for comb in list(combinations(itemsets, k))]
    )


def itemsets_support(transactions, itemsets, min_support):
    support_count = {itemset: 0 for itemset in itemsets}
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                support_count[itemset] += 1
    n_transactions = len(transactions)
    return {itemset: support / n_transactions for itemset, support in support_count.items() if support / n_transactions >= min_support}


def apriori(transactions, min_support):
    items = set(chain(*transactions))
    itemsets = [frozenset([item]) for item in items]
    itemsets_by_length = [set()]
    itemsets_supports = dict()
    k = 1
    while itemsets:
        support_count = itemsets_support(transactions, itemsets, min_support)
        lk = set(support_count.keys())
        itemsets_by_length.append(lk)
        itemsets_supports.update(support_count)
        k += 1
        itemsets = join_set(lk, k)
    frequent_itemsets = set(chain(*itemsets_by_length))
    return frequent_itemsets, itemsets_supports


def association_rules(transactions, min_support, min_confidence):
    frequent_itemsets, itemsets_supports = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        support_itemset = itemsets_supports[itemset]
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            if consequent == frozenset():
                continue
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, confidence))
    return rules






def main():
    # Example usage
    transactions = [
        {1, 3, 4},
        {2, 3, 5},
        {1, 2, 3, 5},
        {2, 5},
    ]
    min_support = 0.5
    min_confidence = 0.7

    rules = association_rules(transactions, min_support, min_confidence)

    for antecedent, consequent, support, confidence in rules:
        print(f"{antecedent} => {consequent} (support={support:.2f}, confidence={confidence:.2f})")


if __name__ == '__main__':
    main()
