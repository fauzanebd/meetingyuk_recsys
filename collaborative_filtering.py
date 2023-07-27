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
    """
    Function to compute stochastic gradient descent
    :param shuffled_training_data: training data for sgd procedure
    :param user_bias_reg: regularization parameter for user bias
    :param place_bias_reg: regularization parameter for place bias
    :param user_reg: regularization parameter for user latent factors
    :param place_reg: regularization parameter for place latent factors
    :param learning_rate: learning rate
    :param p: user latent factors
    :param q: place latent factors
    :param user_bias: user bias
    :param place_bias: place bias
    :param global_bias: global bias
    :return: modified p, q, user_bias, place_bias, global_bias based on sgd
    """
    # We loop through all row in training data
    # Each row contains rating data for a place id by a user id.
    for user_id, place_id, rating_ui in shuffled_training_data.values:
        # We use stochastic gradient descent to optimize the Regularized error function

        # Matrix Factorization principle state that a rating matrix can be factorized into smaller dimension matrix.
        # In this code, rating matrix R factorized into factor matrix P and factor matrix Q
        # So, the ratings by user u for place i is r_ui = P[u] * Q[u]
        # matrix P contains user factors, and Q contains place factors
        # we can get ratings by user id u for place id p by looking at dot product result
        # of user u latent factors and place p latent factors.
        # So, the ratings by user_id for place_id is:
        # r_ui = user_factors[user_id] * place_factors[place_id]

        # But, optimization by Bell and Koren in their solution done by considering biases
        # in the rating prediction.
        # By considering the biases, prediction rating by an user for a place is
        # global bias + user bias + place bias + rating prediction from dot product
        # global bias is the average of all ratings
        # user bias if the average of ratings by current user_id
        # place bias is the average of ratings for current place_id
        # rating prediction is the dot product of user latent factors and place latent factors.
        # So, final equation for rating prediction is:
        # predicted rating = mu + bu + bp + user_factors[user_id] * place_factors[place_id]

        # Then, we aim to minimize squared error function:
        # E = (actual rating - predicted rating)^2
        # To avoid overfitting, we add regularization parameter for biases and latent factors.
        # So, the regularized error function is:
        # E = (actual rating - predicted rating)^2 + lambda * (bu^2 + bp^2 + ||p_u||^2 + ||q_i||^2)
        # In this regularized error, we punish each parameters to be optimized (bias parameter and latent factors parameters) that
        # has high slope, means the parameter had so much power in changing rating predictions
        # how we punish it is by adding squared value of parameters.

        # In this functions, user_factors stored in p variable, and place_factors stored in q variable.
        # We first predict the rating by user_id for place_id, by existing user_factors and place_factors
        # and existing biases.

        # First, we find initial predictions for this loop
        # This code represent mu + bu + bp + user_factors[user_id] * place_factors[place_id]
        prediction = global_bias + user_bias[user_id] + place_bias[place_id] + np.dot(p[user_id], q[place_id])

        # the error is observed rating - predicted rating
        error = rating_ui - prediction

        # Then, the error used to update the parameters.

        # To update user bias
        # New user bias parameter for current looped user_id is updated by adding step size to previous parameter.
        # Step size is derivation from regularized squared error function
        # in respect to user bias bu.
        # step size = learning_rate * error value - learning_rate * regularization for user bias
        step_size = learning_rate * (error - (user_bias_reg * user_bias[user_id]))
        user_bias[user_id] = user_bias[user_id] + step_size

        # To update place bias
        # New place bias parameter for current looped place_id is updated by adding step size to previous parameter
        # Step size is derivation from regularized squared error function
        # in respect to place bias bp
        # step size = learning_rate * error_value - learning_rate * regularization for place bias
        step_size = learning_rate * (error - (place_bias_reg * place_bias[place_id]))
        place_bias[place_id] = place_bias[place_id] + step_size

        # To update user factors p
        # New latent factors for current looped user_id is updated by adding step size to previous latent factors
        # Step size is derivation from regularized squared error function in respect to user factors p
        # step size = learning_rate * error * place factors q - learning_rate * regularization for user factors
        step_size = learning_rate * ((error * q[place_id]) - (user_reg * p[user_id]))
        p[user_id] = p[user_id] + step_size

        # To update place factors q
        # New latent factors for current looped place_id is updated by adding step size to previous latent factors
        # step size is derivation from regularized squared error function in respect to place factors q
        # step size = learning_rate * error * user factors p - learning_rate * regularization for place factors
        step_size = learning_rate * ((error * p[user_id]) - (place_reg * q[place_id]))
        q[place_id] = q[place_id] + step_size

        # If there is another data, we will go on another loop, and place factors will be updated

    # If no training data left, we return the user factors p, place factors q, user bias and place bias.
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
    """
    Function to train the model with SGD
    :param training_rating_data: training data for sgd procedure
    :param user_ids: list of user ids in training data
    :param place_ids: list of place ids in training data
    :param user_bias_reg: regularization parameter for user bias
    :param place_bias_reg: regularization parameter for place bias
    :param user_reg: regularization parameter for user latent factors
    :param place_reg: regularization parameter for place latent factors
    :param number_of_users: number of users in training data
    :param number_of_places: number of places in training data
    :param learning_rate: learning rate used for sgd training
    :param n_epochs: number of epochs used for sgd training
    :param n_factors: number of latent factors
    :return: p, q, user_bias, place_bias, global_bias
    """
    n_user = number_of_users
    n_place = number_of_places

    # Initiate random numbers with normal distribution for user and place factors
    p = dict(zip(
        user_ids, # id of users that exist in training data
        np.random.normal(scale=1. / n_factors, size=(n_user, n_factors)) # initial vector values for an user_id
    ))
    q = dict(zip(
        place_ids, # id of places that exist in training data
        np.random.normal(scale=1. / n_factors, size=(n_place, n_factors)) # initial vector values for a place_id
    ))
    # Initiate biases with zeros
    user_bias = dict(zip(
        user_ids,
        np.zeros(n_user)
    ))
    place_bias = dict(zip(
        place_ids,
        np.zeros(n_place)
    ))
    # global bias or mu is the mean of all ratings
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
    """
    Function to train the model with SGD, the difference from
    train function is, this function does not initiate the random
    factors and biases, so we can call this if there's existing
    factors and biases and we want to continue the training
    using new data.
    In other word, used when already trained data.
    :param training_rating_data: training data for sgd procedure
    :param user_bias_reg: regularization parameter for user bias
    :param place_bias_reg: regularization parameter for place bias
    :param user_reg: regularization parameter for user latent factors
    :param place_reg: regularization parameter for place latent factors
    :param learning_rate: learning rate used for sgd training
    :param n_epochs: number of epochs used for sgd training
    :param p: user latent factors
    :param q: place latent factors
    :param user_bias: user bias
    :param place_bias: place bias
    :param global_bias: global bias
    :return: p, q, user_bias, place_bias, global_bias
    """
    # Loop through epochs
    for _ in range(n_epochs):
        # shufffle the training data
        shuffled_training_data = training_rating_data.sample(frac=1).reset_index(drop=True)
        # call sgd procedure
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
    """
    Function to predict single rating by user_id for place_id
    :param user_id: user id
    :param place_id: place id
    :param p: user latent factors
    :param q: place latent factors
    :param user_bias: user bias
    :param place_bias: place bias
    :param global_bias: global bias
    :return: prediction
    """

    prediction = global_bias + user_bias[user_id] + place_bias[place_id]
    prediction += p[user_id].dot(q[place_id].T)

    return prediction

def run_sgd_background(
    new_rating_data=None,
    learning_rate=0.001,
    regularization=.02,
    n_factors=40,
    iterations=100
):
    """
    Function to run SGD in background, this function will be called
    in background thread so the main flask app thread can still serve the request
    :param new_rating_data: new rating data to be used for online learning, if not defined, training is done with all data exist in db
    :param learning_rate: learning rate
    :param regularization: regularization parameter
    :param n_factors: number of latent factors
    :param iterations: number of iterations/eppochs
    :return: None, the trained user and place factors and user and place and global biases will be saved in db immediately
    """
    print(f"Start calculating SGD")
    user_bias_reg = regularization
    place_bias_reg = regularization
    user_reg = regularization
    place_reg = regularization


    if new_rating_data is None:
        # If this function called without new data specified,
        # we assume the caller wants to train the model with all data
        # get all ratings data from db
        train_data = get_all_ratings()
        train_data = train_data[['user_id', 'place_id', 'rating']]
        p = None
        q = None
        user_bias = None
        place_bias = None
        global_bias = None

    else:
        # If new data specified,
        # we'll gonna do online learning to update existing user and place factors
        length_of_new_data = len(new_rating_data)
        previous_data_length = meetingyuk_mongo_collection('place_db', 'ratings').count_documents({}) - length_of_new_data
        train_data = new_rating_data
        train_user_ids = train_data['user_id'].unique()
        train_place_ids = train_data['place_id'].unique()

        # get user factors from db
        p = get_user_factor_df(train_user_ids)
        p = {col: p[col].values for col in p.columns}

        # get place factors from db
        q = get_place_factor_df(train_place_ids)
        q = {col: q[col].values for col in q.columns}

        # get user bias from db
        user_bias = get_user_bias(train_user_ids)
        user_bias = {key: [val] for key, val in user_bias.items()}

        # get place bias from db
        place_bias = get_place_bias(train_place_ids)
        place_bias = {key: [val] for key, val in place_bias.items()}

        # get global bias from db
        existing_global_bias = get_global_bias()

        # calculate new global bias based on new data
        global_bias = (
            previous_data_length * existing_global_bias + np.sum(train_data['rating'])
        ) / (previous_data_length + length_of_new_data)


    # map the ids to integers, this is needed to index the factors matrix
    uid_to_int = {uid: iid for iid, uid in enumerate(train_data['user_id'].unique())}
    pid_to_int = {pid: iid for iid, pid in enumerate(train_data['place_id'].unique())}
    int_to_uid = {iid: uid for iid, uid in enumerate(train_data['user_id'].unique())}
    int_to_pid = {iid: pid for iid, pid in enumerate(train_data['place_id'].unique())}

    train_data['user_id'] = train_data['user_id'].map(uid_to_int)
    train_data['place_id'] = train_data['place_id'].map(pid_to_int)
    user_ids_int = train_data['user_id'].unique()
    place_ids_int = train_data['place_id'].unique()

    print(f"Start training SGD")
    # If there's existing factors, we'll do online learning
    if p is not None:
        p, q, user_bias, place_bias, global_bias = partial_train(
            train_data, user_bias_reg, place_bias_reg,
            user_reg, place_reg, learning_rate, iterations,
            p, q, user_bias, place_bias, global_bias
        )
    # If there's no existing factors, we learn from scratch
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


