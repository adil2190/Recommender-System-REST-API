from flask import Flask, json, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from firebase_admin import credentials, firestore, initialize_app


# Init app
app = Flask(__name__)


# Initialize firestore
cred = credentials.Certificate("./serviceAccountKey.json")
default_app = initialize_app(cred)
db = firestore.client()
products_ref = db.collection('Products').stream()

productsArr = []


def mergeDict(items):
    newArr = []
    print(items)
    for item in items:
        newArr.append({k: v for x in item for k, v in x.items()})

    return newArr


def mergeDict1(items):
    newArr = []
    # print(items)
    newArr.append({k: v for x in items for k, v in x.items()})

    return newArr


for doc in products_ref:
    productsArr.append(doc.to_dict())
    # print(doc.to_dict())

products_df = pd.DataFrame(productsArr)
specs_df = products_df[['Specs', 'Productname']]
# specs_list = products_df['Specs'].to_list()
# specs_converted = mergeDict(specs_list)
# specs_df = pd.DataFrame(specs_converted)

specs_df['Specs'] = specs_df['Specs'].apply(lambda x: mergeDict1(x))
print(specs_df)

laptops = pd.read_csv('laptops.csv', encoding='latin-1')
laptops = laptops[['id', 'Company', 'Product', 'TypeName',
                   'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Price_euros']]
laptops['Price_euros'] = laptops['Price_euros'].apply(lambda x: f'price{x}')
laptops['Ram'] = laptops['Ram'].apply(lambda x: f'RAM{x}')
laptops['Cpu'] = laptops['Cpu'].apply(lambda x: x.replace(' ', ''))
laptops['Gpu'] = laptops['Gpu'].apply(lambda x: x.replace(' ', ''))
laptops['Memory'] = laptops['Memory'].apply(lambda x: x.replace(' ', ''))
laptops['OpSys'] = laptops['OpSys'].apply(lambda x: x.replace(' ', ''))
laptops['tags'] = laptops['Company'] + ' ' + laptops['Cpu'] + ' ' + laptops['Memory'] + ' ' + laptops['OpSys'] + \
    ' ' + laptops['Ram'] + ' ' + laptops['TypeName'] + \
    ' ' + laptops['Price_euros'] + ' ' + laptops['Gpu']
new_df = laptops[['id', 'Product', 'tags']]

cv = CountVectorizer(max_features=520)
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)


def recommend(item):
    # print(item)
    item_index = new_df[new_df['id'] == item].index[0]
    distances = similarity[item_index]
    item_list = sorted(list(enumerate(distances)),
                       reverse=True, key=lambda x: x[1])[1:6]
    return item_list


def formatted_results(result):
    arr = []
    for r in result:
        my_dict = {}
        my_dict['product_index'] = r[0]
        my_dict['similarity_score'] = r[1]
        arr.append(my_dict)
    return (arr)


def detailed_results(result):
    arr = []
    for r in result:
        my_dict = {}
        my_dict['similarity_score'] = r[1]
        my_dict['product_name'] = new_df.iloc[r[0]].Product
        my_dict['specifications'] = new_df.iloc[r[0]].tags
        arr.append(my_dict)
    return (arr)


# test
@app.route('/', methods=['GET'])
def get():
    result = recommend(16)
    new_result = formatted_results(result)
    return jsonify({'Result': new_result})


@app.route('/contentBasedRecommendation', methods=['GET'])
def content_based_recommendation():
    input_product = int(request.args.get('product'))
    result = recommend(input_product)
    new_result = detailed_results(result)
    return jsonify({'Result': new_result})


@app.route('/addMessage', methods=['POST'])
def message():
    message = request.json['message']
    print(message)
    return jsonify({'response': productsArr})

# Run server


if __name__ == '__main__':
    app.run(debug=True)
