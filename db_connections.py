import pandas as pd
from pymongo import MongoClient
from pymongo import UpdateOne
import os

def meetingyuk_mongo_collection(db_name, collection_name):
    client = MongoClient(
        os.getenv('MONGODB_URI')
    )
    db = client[db_name]
    collection = db[collection_name]
    return collection

def meetingyuk_mongo_db(db_name):
    client = MongoClient(
        os.getenv('MONGODB_URI')
    )
    db = client[db_name]
    return db

def get_all_merchants():
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find())
    dataset = pd.json_normalize(dataset)
    return dataset

def get_merchant_details(merchant_ids):
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({'_id': {'$in': merchant_ids}}, {'_id': 1, 'name': 1, 'address': 1, 'opening_hours': 1, 'image_url': 1, 'rooms': 1, 'ratings': 1, 'review_count': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_all_merchant_ids():
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({}, {'_id': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset['_id'].tolist()

def get_all_merchant_id_and_locs():
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({}, {'_id': 1, 'location': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_all_ratings():
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find())
    dataset = pd.json_normalize(dataset)
    return dataset

def get_user_rated_places(user_id):
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find({'user_id': user_id}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_new_ratings_data(data_ids):
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find({'_id': {'$in': data_ids}}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_user_factor_df(user_ids):
    collection = meetingyuk_mongo_collection('recsys', 'user_factors')
    if not isinstance(user_ids, list):
        user_ids = [user_ids]
    data = collection.find({"entity_id": {"$in": user_ids}})
    data = list(data)
    data = pd.json_normalize(data)
    data = data.set_index('entity_id').T
    data.drop(['_id'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data

def get_place_factor_df(place_ids):
    collection = meetingyuk_mongo_collection('recsys', 'place_factors')
    if not isinstance(place_ids, list):
        place_ids = [place_ids]
    data = collection.find({"entity_id": {"$in": place_ids}})
    data = list(data)
    data = pd.json_normalize(data)
    data = data.set_index('entity_id').T
    data.drop(['_id'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data

def get_user_bias(user_ids):
    collection = meetingyuk_mongo_collection('recsys', 'user_bias')
    if not isinstance(user_ids, list):
        user_ids = [user_ids]
    data = collection.find({"entity_id": {"$in": user_ids}})
    data = list(data)
    data = pd.json_normalize(data)
    data = data.set_index('entity_id').T
    data.drop(['_id'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = data.explode(data.columns.tolist())
    data = data.to_dict('records')[0]
    return data

def get_place_bias(place_ids):
    collection = meetingyuk_mongo_collection('recsys', 'place_bias')
    if not isinstance(place_ids, list):
        place_ids = [place_ids]
    data = collection.find({"entity_id": {"$in": place_ids}})
    data = list(data)
    data = pd.json_normalize(data)
    data = data.set_index('entity_id').T
    data.drop(['_id'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = data.explode(data.columns.tolist())
    data = data.to_dict('records')[0]
    return data


def get_global_bias():
    collection = meetingyuk_mongo_collection('recsys', 'global_recsys_config')
    global_bias_entry = collection.find_one({"type": "global_bias"})
    if global_bias_entry:
        return global_bias_entry.get('value', None)
    else:
        return None


def store_dataframe_to_mongo(df, collection, factor_key):
    operations = []

    for col in df.columns:
        doc = {
            "entity_id": col,
            factor_key: df[col].tolist()
        }
        # Prepare update operation
        operation = UpdateOne({"entity_id": col}, {"$set": doc}, upsert=True)
        operations.append(operation)

    # Execute all operations in a single batch
    if operations:
        collection.bulk_write(operations)