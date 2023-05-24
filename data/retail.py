import pandas as pd
import random, math


def create_transactions_and_users():
    data = open('retail.dat')
    lines = data.readlines()[1:]
    transactions = []
    test_data = []

    for l in lines:
        items = set(map(int, l.split()))
        if len(items) > 10:
            if random.random() < 0.3:
                items_user = set(random.sample(items, math.floor(len(items)/2)))
                true_items = items - items_user
                test_data.append([items_user, true_items])
            else:
                transactions.append(items)
        else:
            transactions.append(items)

    print(f'Extracted {len(transactions)} transactions and {len(test_data)} users from total {len(lines)} rows')

    test_data = pd.DataFrame(test_data, columns=['items_user', 'true_items'])
    test_data.index.name = "ID"
    test_data.to_csv("retail-users.csv")

    transactions = pd.DataFrame({"transaction": transactions}, columns=['transaction'])
    transactions.index.name = "ID"
    transactions.to_csv("retail-transactions.csv")


if __name__ == '__main__':
    random.seed(42)
    create_transactions_and_users()