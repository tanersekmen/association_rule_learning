import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Quest 1: Data preprocessing.
df_first = pd.read_excel("online_retail_II.xlsx" , sheet_name="Year 2010-2011")
df = df_first.copy()
print(df.head())
print(df.shape)
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

#Quest 2: Build association rule from Germany customers.

germany_df = df[df['Country'] == 'Germany']
germany_df.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:10, 0:10]

# changing the values of all product between 0 and 1 according to Invoice.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().\
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().\
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# sees items in the columns, invoices in the rows
# invoice-product matrix
inv_germany_df = create_invoice_product_df(germany_df)


inv_germany_df = create_invoice_product_df(germany_df, id=True)

# number of frequencies of items and their support ratio
frequent_itemsets = apriori(inv_germany_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(20)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(20)


# Quest 3: What are the names of the products whose IDs are given?

# User 1, product id: 21987
# User 2, product id: 23235
# User 3, product id: 22747
def id_finder(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)



id_finder(germany_df,21987)
id_finder(germany_df,23235)
id_finder(germany_df,22747)



# Quest 4 : Make a product recommendation for users in the cart.
def arl_recommender(df_rules, product_id, rec_count=1):

    sorted_rules = df_rules.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)



# Quest 5: What is the names of product that algorithm recommended?

product_names = []
product_names += arl_recommender(rules, 21987, 1)
product_names += arl_recommender(rules, 23235, 2)
product_names += arl_recommender(rules, 22747, 3)

for i in product_names:
    id_finder(germany_df,i)
    #--results--
    #['SET OF 60 PANTRY DESIGN CAKE CASES ']
    #['RED RETROSPOT MINI CASES']
    #['PLASTERS IN TIN WOODLAND ANIMALS']


