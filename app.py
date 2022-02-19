from math import prod
from flask import Flask, json, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from firebase_admin import credentials, firestore, initialize_app


# Init app
app = Flask(__name__)


# Initialize firestore
cred = credentials.Certificate("./serviceAccountKey.json")
default_app = initialize_app(cred)
db = firestore.client()

# def formatted_results(result):
#     arr = []
#     for r in result:
#         my_dict = {}
#         my_dict['product_index'] = r[0]
#         my_dict['similarity_score'] = r[1]
#         arr.append(my_dict)
#     return (arr)


# laptops = pd.read_csv('laptops.csv', encoding='latin-1')
# laptops = laptops[['id', 'Company', 'Product', 'TypeName',
#                    'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Price_euros']]
# laptops['Price_euros'] = laptops['Price_euros'].apply(lambda x: f'price{x}')
# laptops['Ram'] = laptops['Ram'].apply(lambda x: f'RAM{x}')
# laptops['Cpu'] = laptops['Cpu'].apply(lambda x: x.replace(' ', ''))
# laptops['Gpu'] = laptops['Gpu'].apply(lambda x: x.replace(' ', ''))
# laptops['Memory'] = laptops['Memory'].apply(lambda x: x.replace(' ', ''))
# laptops['OpSys'] = laptops['OpSys'].apply(lambda x: x.replace(' ', ''))
# laptops['tags'] = laptops['Company'] + ' ' + laptops['Cpu'] + ' ' + laptops['Memory'] + ' ' + laptops['OpSys'] + \
#     ' ' + laptops['Ram'] + ' ' + laptops['TypeName'] + \
#     ' ' + laptops['Price_euros'] + ' ' + laptops['Gpu']
# new_df = laptops[['id', 'Product', 'tags']]

# print(new_df)


# adding the recommended products in the buyers subcollection
def findProducts(ids, userId):
    print(userId)
    arr = []
    count = 1
    for id in ids:
        doc_ref = db.collection('Products').document(id)
        doc = doc_ref.get()
        myDict = doc.to_dict()
        myDict['index'] = count
        print(myDict)
        if doc.exists:
            arr.append(myDict)
            db.collection('Buyers').document(userId).collection(
                'ContentRecommended').document(doc.id).set(myDict)
        count = count + 1
    print('success')


def recommend(item, userId):
    # print(item)
    products_ref = db.collection('Products').stream()
    productsArr = []

# getting the collection of products from firestore
    for doc in products_ref:
        mydict = doc.to_dict()
        mydict['productId'] = doc.id
        productsArr.append(mydict)
        # print(doc.to_dict())

    products_df = pd.DataFrame(productsArr)

    products_df = products_df[['Productname',
                               'Price', 'productId', 'Specs', 'Category']]

    products_df.rename(columns={'Productname': 'productName', 'Price': 'price',
                                'productId': 'id', 'Specs': 'specs', 'Category': 'category'}, inplace=True)

    products_df['price'] = products_df['price'].apply(lambda x: f'price{x}')
    products_df['tags'] = products_df['specs'] + ' ' + \
        products_df['price'] + ' ' + products_df['category'] + \
        ' ' + products_df['productName']
    # print(products_df['tags'])
    final_products_df = products_df[['id', 'productName', 'tags']]
    cv = CountVectorizer(max_features=520)
    vectors = cv.fit_transform(final_products_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    item_index = final_products_df[final_products_df['id'] == item].index[0]
    distances = similarity[item_index]
    item_list = sorted(list(enumerate(distances)),
                       reverse=True, key=lambda x: x[1])[1:6]

    print(item_list)
    arr = []
    for r in item_list:
        arr.append(final_products_df.iloc[r[0]].id)
    findProducts(arr, userId)
    return (arr)


def collaborative_recommend(item):
    ratings_ref = db.collection('Ratings').stream()
    ratingsArr = []

    for doc in ratings_ref:
        ratingsArr.append(doc.to_dict())

    ratings_df = pd.DataFrame(ratingsArr)

    ratings_pivot = ratings_df.pivot_table(
        columns='userId', index='productId', values='rating')
    ratings_pivot.fillna(0, inplace=True)

    ratings_sparse = csr_matrix(ratings_pivot)
    model = NearestNeighbors(algorithm='brute')
    model.fit(ratings_sparse)
    # print('item --------> ', np.where(ratings_pivot.index == item)[0][0])
    try:
        input_index = np.where(ratings_pivot.index == item)[0][0]
        print(input_index)
        distances, suggestions = model.kneighbors(
            ratings_pivot.iloc[input_index, :].values.reshape(1, -1), n_neighbors=2)
        arr = []
        for item in suggestions[0]:
            arr.append(ratings_pivot.index[item])

        return arr
    except:
        return []


# def detailed_results(result):


# test
@app.route('/', methods=['GET'])
def get():
    return jsonify({'Result': "Hello World"})


@app.route('/contentBasedRecommendation', methods=['GET'])
def content_based_recommendation():
    input_product = request.args.get('product')
    user_id = request.args.get('userId')
    result = recommend(input_product, user_id)
    # new_result = detailed_results(result)
    return jsonify({'Result': result})


@app.route('/collaborativeRecommendation', methods=['GET'])
def collaborative_filtering():
    input_product = request.args.get('product')
    result = collaborative_recommend(input_product)[1:6]
    return jsonify({'Result': result})

# Run server


if __name__ == '__main__':
    app.run(debug=True)
