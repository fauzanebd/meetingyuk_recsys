import numpy as np
import pandas as pd
from pandas import json_normalize

from db_connections import get_all_ratings, meetingyuk_mongo_collection, get_user_factor_df, get_place_factor_df, \
    get_user_bias, get_place_bias, get_global_bias, store_dataframe_to_mongo


def compute_sgd(
    shuffled_training_data,
    user_bias_reg, place_bias_reg, user_reg,
    place_reg, learning_rate, p, q,
    user_bias, place_bias, global_bias
):
    for user_id, place_id, rating_ui in shuffled_training_data.values:
        prediction = global_bias + user_bias[user_id] + place_bias[place_id]
        prediction += np.dot(p[user_id], q[place_id])
        error = rating_ui - prediction

        # Update biases
        user_bias[user_id] += learning_rate * (error - (user_bias_reg * user_bias[user_id]))
        place_bias[place_id] += learning_rate * (error - (place_bias_reg * place_bias[place_id]))

        # Update latent factors
        p_u = p[user_id]
        p[user_id] += learning_rate * ((error * q[place_id]) - (user_reg * p[user_id]))
        q[place_id] += learning_rate * ((error * p_u) - (place_reg * q[place_id]))

    return p, q, user_bias, place_bias


def train(
    training_rating_data: pd.DataFrame,
    user_ids,
    place_ids,
    user_bias_reg,
    place_bias_reg,
    user_reg,
    place_reg,
    number_of_users: int,
    number_of_places: int,
    learning_rate=.01,
    n_epochs=10,
    n_factors=10
):
    n_user = number_of_users
    n_place = number_of_places
    p = dict(zip(
        user_ids,
        np.random.normal(scale=1. / n_factors, size=(n_user, n_factors))
    ))
    q = dict(zip(
        place_ids,
        np.random.normal(scale=1. / n_factors, size=(n_place, n_factors))
    ))
    user_bias = dict(zip(
        user_ids,
        np.zeros(n_user)
    ))
    place_bias = dict(zip(
        place_ids,
        np.zeros(n_place)
    ))
    global_bias = np.mean(training_rating_data['rating'])
    p, q, user_bias, place_bias, global_bias = partial_train(
        training_rating_data, user_bias_reg, place_bias_reg,
        user_reg, place_reg, learning_rate, n_epochs,
        p, q, user_bias, place_bias, global_bias
    )
    return p, q, user_bias, place_bias, global_bias

def partial_train(
        training_rating_data, user_bias_reg, place_bias_reg,
        user_reg, place_reg, learning_rate, n_epochs,
        p, q, user_bias, place_bias, global_bias
):
    for _ in range(n_epochs):
        shuffled_training_data = training_rating_data.sample(frac=1).reset_index(drop=True)
        # print(f"length of training indices: {len(shuffled_training_data)}")
        p, q, user_bias, place_bias = compute_sgd(
            shuffled_training_data,
            user_bias_reg, place_bias_reg, user_reg,
            place_reg, learning_rate, p, q,
            user_bias, place_bias, global_bias
        )
    return p, q, user_bias, place_bias, global_bias

def predict_single(
    user_id,
    place_id,
    p,
    q,
    user_bias,
    place_bias,
    global_bias
):
    prediction = global_bias + user_bias[user_id] + place_bias[place_id]
    prediction += p[user_id].dot(q[place_id].T)

    return prediction

def run_sgd_background(
    new_rating_data=None,
    learning_rate=0.001,
    regularization=.02,
    n_factors=40,
    iterations=1
):
    print(f"Start calculating SGD")
    user_bias_reg = regularization
    place_bias_reg = regularization
    user_reg = regularization
    place_reg = regularization


    if new_rating_data is None:
        # get all ratings data from db
        train_data = get_all_ratings()
        train_data = train_data[['user_id', 'place_id', 'rating']]
        p = None
        q = None
        user_bias = None
        place_bias = None
        global_bias = None

    else:
        length_of_new_data = len(new_rating_data)
        previous_data_length = meetingyuk_mongo_collection('place_db', 'ratings').count_documents({}) - length_of_new_data
        train_data = new_rating_data
        train_user_ids = train_data['user_id'].unique()
        train_place_ids = train_data['place_id'].unique()
        p = get_user_factor_df(train_user_ids)
        p = {col: p[col].values for col in p.columns}

        q = get_place_factor_df(train_place_ids)
        q = {col: q[col].values for col in q.columns}

        user_bias = get_user_bias(train_user_ids)
        user_bias = {key: [val] for key, val in user_bias.items()}

        place_bias = get_place_bias(train_place_ids)
        place_bias = {key: [val] for key, val in place_bias.items()}

        existing_global_bias = get_global_bias()

        global_bias = (
            previous_data_length * existing_global_bias + np.sum(train_data['rating'])
        ) / (previous_data_length + length_of_new_data)

    uid_to_int = {uid: iid for iid, uid in enumerate(train_data['user_id'].unique())}
    pid_to_int = {pid: iid for iid, pid in enumerate(train_data['place_id'].unique())}
    int_to_uid = {iid: uid for iid, uid in enumerate(train_data['user_id'].unique())}
    int_to_pid = {iid: pid for iid, pid in enumerate(train_data['place_id'].unique())}

    train_data['user_id'] = train_data['user_id'].map(uid_to_int)
    train_data['place_id'] = train_data['place_id'].map(pid_to_int)
    user_ids_int = train_data['user_id'].unique()
    place_ids_int = train_data['place_id'].unique()

    print(f"Start training SGD")
    if p is not None:
        p, q, user_bias, place_bias, global_bias = partial_train(
            train_data, user_bias_reg, place_bias_reg,
            user_reg, place_reg, learning_rate, iterations,
            p, q, user_bias, place_bias, global_bias
        )
    else:
        p, q, user_bias, place_bias, global_bias = train(
            train_data, user_ids_int, place_ids_int,
            user_bias_reg, place_bias_reg, user_reg,
            place_reg, len(user_ids_int), len(place_ids_int),
            learning_rate=learning_rate,
            n_epochs=iterations,
            n_factors=n_factors
        )

    p_df = pd.DataFrame(p)
    q_df = pd.DataFrame(q)
    ubidf = pd.DataFrame(user_bias, index=[0])
    pbidf = pd.DataFrame(place_bias, index=[0])

    p_df.columns = [int_to_uid[i] for i in p_df.columns]
    q_df.columns = [int_to_pid[i] for i in q_df.columns]
    ubidf.columns = [int_to_uid[i] for i in ubidf.columns]
    pbidf.columns = [int_to_pid[i] for i in pbidf.columns]

    print('storing user factors')
    store_dataframe_to_mongo(
        p_df,
        meetingyuk_mongo_collection('real_recsys', 'user_factors'),
        'latent_factors'
    )

    print('storing place factors')
    store_dataframe_to_mongo(
        q_df,
        meetingyuk_mongo_collection('real_recsys', 'place_factors'),
        'latent_factors'
    )

    print('storing user bias')
    store_dataframe_to_mongo(
        ubidf,
        meetingyuk_mongo_collection('real_recsys', 'user_bias'),
        'bias'
    )

    print('storing place bias')
    store_dataframe_to_mongo(
        pbidf,
        meetingyuk_mongo_collection('real_recsys', 'place_bias'),
        'bias'
    )

    print('storing global bias')
    meetingyuk_mongo_collection('real_recsys', 'global_recsys_config').update_one(
        {'type': 'global_bias'},
        {'$set': {'value': global_bias}},
        upsert=True
    )

    print(f"Finished calculating SGD")


