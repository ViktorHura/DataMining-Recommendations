import pandas as pd
import math
import pickle
import functools
from recommender import evaluate_recommendations, recommend_items_custom_metric, Jaccard_metric, Imbalance_Ratio_metric
from ndi_python import association_rules

CACHED = True

def main():
    min_support = 0.0025

    print(f'min_support: {min_support}')

    transactions = pd.read_csv("../../data/retail2-transactions.tsv", sep="\t", index_col=0)
    # yes the eval looks sketchy, it is to convert strings to sets
    transactions = [i for i in transactions['transaction'].apply(eval).tolist()]

    confidences = [0.6, 0.7, 0.8, 0.9]
    rule_list = []

    for c in confidences:
        print(f"== Generating association rules with min_support: {min_support} "
              f"(≈{math.floor(min_support*len(transactions))}/{len(transactions)} transaction) and min_confidence: {c}")
        print()

        if CACHED:
            with open(f'cache/rulesNDI{c}.pickle', 'rb') as handle:
                rules = pickle.load(handle)
        else:
            rules = association_rules(transactions, min_support, c)
            pickle.dump(rules, open(f"cache/rulesNDI{c}.pickle", "wb"))

        rule_list.append(rules)
        print(f'Generated {len(rules)} rules \n')

    # prepare the test user data into two dictionaries
    test_data = pd.read_csv("../../data/retail2-users.tsv", sep="\t", index_col=0)
    user_inputs = dict()
    user_true_items = dict()
    for index, row in test_data.iterrows():
        user_inputs[index] = eval(row[0])
        user_true_items[index] = eval(row[1])

    top_ks = [1, 3, 5, 7, 10, 15, 20]
    confidences = [0.6, 0.7, 0.8, 0.9]

    metric_results_dict = {c: None for c in confidences}

    def make_results_dic(k):
        d = metric_results_dict.copy()
        d.update({'K': k})
        return d

    data = {
        'precision': [make_results_dic(k) for k in top_ks],
        'recall': [make_results_dic(k) for k in top_ks],
        'F1': [make_results_dic(k) for k in top_ks]
    }

    for i, c in enumerate(confidences):
        for j, k in enumerate(top_ks):
            print(f'Evaluating top_k={k}, confidence: {c}')
            result = evaluate_recommendations(functools.partial(recommend_items_custom_metric, metric_index=3, method='max'),
                                       user_true_items, user_inputs, rule_list[i], k)

            data['precision'][j][c] = result[0]
            data['recall'][j][c] = result[1]
            data['F1'][j][c] = result[2]

    pickle.dump(data, open("cache/dataNDI.pickle", "wb"))
    print()


if __name__ == '__main__':
    main()