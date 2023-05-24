import pandas as pd
import random, math


def create_transactions_and_users():
    data = pd.read_csv('UCI-retail.csv')
    data.drop(['Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country'], axis=1, inplace=True)

    transactions = []
    test_data = []

    grouped = data.groupby(['Invoice'])
    for name, group in grouped:
        items = group['StockCode'].to_list()
        if len(items) > 10:
            if random.random() < 0.2:
                items_user = set(random.sample(items, math.floor(len(items) / 2)))
                true_items = set(items) - items_user
                test_data.append([items_user, true_items])
            else:
                transactions.append([set(items)])
        else:
            transactions.append([set(items)])

    print(f'Extracted {len(transactions)} transactions and {len(test_data)} users from total {len(grouped)} rows')

    test_data = pd.DataFrame(test_data, columns=['items_user', 'true_items'])
    test_data = test_data.astype('object')
    test_data.index.name = "ID"
    test_data.to_csv("retail2-users.tsv", sep="\t")

    transactions = pd.DataFrame(transactions, columns=['transaction'])
    transactions = transactions.astype('object')
    transactions.index.name = "ID"
    transactions.to_csv("retail2-transactions.tsv", sep="\t")


if __name__ == '__main__':
    random.seed(42)
    create_transactions_and_users()