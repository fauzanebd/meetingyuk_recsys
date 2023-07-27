import pandas as pd
from pymongo import MongoClient
from pymongo import UpdateOne
import os

def meetingyuk_mongo_collection(db_name, collection_name):
    """
    Get MongoDB collection
    :param db_name: get database name
    :param collection_name: get collection name
    :return:
    """
    client = MongoClient(
        os.getenv('MONGODB_URI')
    )
    db = client[db_name]
    collection = db[collection_name]
    return collection

def meetingyuk_mongo_db(db_name):
    """
    Get MongoDB database
    :param db_name: get database name
    :return: mongodb database
    """
    client = MongoClient(
        os.getenv('MONGODB_URI')
    )
    db = client[db_name]
    return db

def get_all_merchants():
    """
    Get all merchants from MongoDB
    :return: all merchants that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find())
    dataset = pd.json_normalize(dataset)
    return dataset

def get_merchant_details(merchant_ids):
    """
    Get merchant details from MongoDB
    :param merchant_ids: list of merchant ids
    :return: merchant details for given merchant ids
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({'_id': {'$in': merchant_ids}}, {'_id': 1, 'name': 1, 'address': 1, 'opening_hours': 1, 'image_url': 1, 'rooms': 1, 'ratings': 1, 'review_count': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_all_merchant_ids():
    """
    Get all merchant ids from MongoDB
    :return: all merchant ids that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({}, {'_id': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset['_id'].tolist()

def get_all_merchant_id_and_locs():
    """
    Get all merchant id and location from MongoDB
    :return: all merchant id and location (in latitude and longitude degree) that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.places.find({}, {'_id': 1, 'location': 1}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_all_ratings():
    """
    Get all ratings from MongoDB
    :return: all ratings that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find())
    dataset = pd.json_normalize(dataset)
    return dataset

def get_user_rated_places(user_id):
    """
    Get user rated places from MongoDB
    :param user_id: user id
    :return: places rated by given user id that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find({'user_id': user_id}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_new_ratings_data(data_ids):
    """
    Get new ratings data from MongoDB
    :param data_ids: list of data ids
    :return: new ratings data for each data id that stored in MongoDB
    """
    db = meetingyuk_mongo_db('place_db')
    dataset = list(db.ratings.find({'_id': {'$in': data_ids}}))
    dataset = pd.json_normalize(dataset)
    return dataset

def get_user_factor_df(user_ids):
    """
    Get user factor dataframe
    :param user_ids: list of user ids
    :return: user factor for each user id that stored in MongoDB
    """
    collection = meetingyuk_mongo_collection('real_recsys', 'user_factors')
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
    """
    Get place factor dataframe
    :param place_ids: list of place ids
    :return: place factor for each place id that stored in MongoDB
    """
    collection = meetingyuk_mongo_collection('real_recsys', 'place_factors')
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
    """
    Get user bias value
    :param user_ids: list of user ids
    :return: user bias for each user id that stored in MongoDB
    """
    collection = meetingyuk_mongo_collection('real_recsys', 'user_bias')
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
    """
    Get place bias value
    :param place_ids: list of place ids
    :return: place bias for each place id that stored in MongoDB
    """
    collection = meetingyuk_mongo_collection('real_recsys', 'place_bias')
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
    """
    Get global bias value
    :return: global bias value that stored in MongoDB
    """
    collection = meetingyuk_mongo_collection('real_recsys', 'global_recsys_config')
    global_bias_entry = collection.find_one({"type": "global_bias"})
    if global_bias_entry:
        return global_bias_entry.get('value', None)
    else:
        return None


def store_dataframe_to_mongo(df, collection, factor_key):
    """
    Store a dataframe to MongoDB
    :param df: dataframe to store
    :param collection: MongoDB collection
    :param factor_key: factor key name  (e.g. "latent_factors")
    :return:
    """
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