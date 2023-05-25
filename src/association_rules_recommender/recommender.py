import pandas as pd
import math
import pickle
from efficient_apriori import apriori
import functools


def evaluate_recommendations(recommend_func, user_true_items, user_inputs, rules, top_n=5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for user, true_items in user_true_items.items():
        # Assuming user_items is a dictionary with user IDs as keys and their associated items as values
        input_items = user_inputs[user]
        # Get recommendations for the user
        recommended_items = set(recommend_func(input_items, rules, top_n=top_n))
        true_items = set(true_items)
        true_positives += len(recommended_items.intersection(true_items))
        false_positives += len(recommended_items - true_items)
        false_negatives += len(true_items - recommended_items)
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def recommend_items_unranked(input_items, rules):
    recommendations = {}
    for r in rules:
        antecedent = r[0]
        consequent = r[1]
        if antecedent.issubset(input_items) and not consequent.issubset(input_items):
            for item in consequent:
                if item not in input_items:
                    if item not in recommendations:
                        recommendations[item] = []
                    recommendations[item].append(r)
    return recommendations


def recommend_items_custom_metric(input_items, rules, metric_index, method, top_n):
    recommendations = recommend_items_unranked(input_items, rules)
    if method == 'max':
        recommendations = {
            item: max(m[metric_index] for m in item_rules)
            for item, item_rules in recommendations.items()
        }
    elif method == 'avg':
        recommendations = {
            item: sum(m[metric_index] for m in item_rules) / len(item_rules)
            for item, item_rules in recommendations.items()
        }
    elif method == 'wmx':
        recommendations = {
            item: max(m[metric_index] for m in item_rules) * (1 - (1/len(item_rules)))
            for item, item_rules in recommendations.items()
        }
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_recommendations[:top_n]]


def Jaccard_metric(rule):
    return rule.support / ( (rule.count_lhs/ rule.num_transactions) + (rule.count_rhs / rule.num_transactions) - rule.support)


def Imbalance_Ratio_metric(rule):
    suppX = (rule.count_lhs/ rule.num_transactions)
    suppY = (rule.count_rhs / rule.num_transactions)
    return abs(suppX - suppY) / ( suppX + suppY - rule.support)


def main():
    min_support = 0.0025
    min_confidence = 0.6

    print(f'min_support: {min_support}, min_confidence: {min_confidence}')

    transactions = pd.read_csv("../../data/retail2-transactions.tsv", sep="\t", index_col=0)
    # yes the eval looks sketchy, it is to convert strings to sets
    transactions = [i for i in transactions['transaction'].apply(eval).tolist()]

    print(f"== Generating association rules with min_support: {min_support} "
          f"(≈{math.floor(min_support*len(transactions))}/{len(transactions)} transaction) and min_confidence: {min_confidence} ==")
    print()

    # generate association rules using a fast apriori implementation
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

    rules = [(frozenset(r.lhs), frozenset(r.rhs), r.support,
              r.confidence, r.lift, r.conviction, r.rpf,
              Jaccard_metric(r), Imbalance_Ratio_metric(r)) for r in rules]

    pickle.dump(rules, open("rules.pickle", "wb"))
    print(f'Generated {len(rules)} rules \n')

    # prepare the test user data into two dictionaries
    test_data = pd.read_csv("../../data/retail2-users.tsv", sep="\t", index_col=0)
    user_inputs = dict()
    user_true_items = dict()
    for index, row in test_data.iterrows():
        user_inputs[index] = eval(row[0])
        user_true_items[index] = eval(row[1])

    # metric_index - metric
    # 3 - confidence
    # 4 - lift
    # 5 - Conviction
    # 6 - Rule Power Factor
    # 7 - Jaccard Coefficient
    # 8 - Imbalance Ratio
    metric_names = ['Confidence', 'Lift', 'Conviction', 'Rule Power Factor', 'Jaccard Coefficient', 'Imbalance Ratio']

    top_ks = [1, 3, 5, 7, 10, 15, 20]

    metric_results_dict = {(name + '_avg'): None for name in metric_names}
    metric_results_dict.update({(name + '_max'): None for name in metric_names})

    def make_results_dic(k):
        d = metric_results_dict.copy()
        d.update({'K': k})
        return d

    data = {
        'precision': [make_results_dic(k) for k in top_ks],
        'recall': [make_results_dic(k) for k in top_ks],
        'F1': [make_results_dic(k) for k in top_ks]
    }

    for j,k in enumerate(top_ks):
        for i in range(3, 9):
            metric_name = metric_names[i - 3]
            for method in ['avg', 'max', 'wmx']:
                print(f'Evaluating top_k={k}, metric={metric_name}, method={method}')
                result = evaluate_recommendations(functools.partial(recommend_items_custom_metric, metric_index=i, method=method),
                                           user_true_items, user_inputs, rules, k)

                metric_name_method = metric_name + '_' + method

                data['precision'][j][metric_name_method] = result[0]
                data['recall'][j][metric_name_method] = result[1]
                data['F1'][j][metric_name_method] = result[2]

    pickle.dump(data, open("data.pickle", "wb"))
    print()


if __name__ == '__main__':
    main()