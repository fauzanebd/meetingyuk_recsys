from db_connections import get_user_rated_places, get_all_merchant_ids, get_place_factor_df, get_user_factor_df, \
    get_user_bias, get_place_bias, get_global_bias, get_merchant_details
from nearest_recommendation import nearest_recommendation
import numpy as np
import pandas as pd


def for_you_recommendation(user_id, max_returns=10000, user_latitude=None, user_longitude=None, include_rated=False, max_radius=10000):
    """
    This function returns recommendations for a user based on his/her location and previous ratings
    :param user_id: user id
    :param max_returns: maximum number of recommendations to return
    :param user_latitude: user latitude
    :param user_longitude: user longitude
    :param include_rated: option to include rated places in recommendation
    :param max_radius: maximum radius to search for nearest places
    :return: json string of recommendations
    """


    if user_latitude and user_longitude:
        # If user location defined, get nearest merchants
        try:
            nearest_places = nearest_recommendation(user_latitude, user_longitude, max_radius=max_radius, from_api=False)
        except ValueError:
            raise ValueError("No recommendations found within the radius of {} km".format(max_radius))
        recommendation_candidate_places_ids = nearest_places['_id'].tolist()
        # get user rated places
        user_rated_places = get_user_rated_places(user_id)
        # option to include rated place in recommendation
        if not include_rated:
            recommendation_candidate_places_ids = list(set(recommendation_candidate_places_ids) - set(user_rated_places['place_id'].tolist()))
    else:
        # If user location not defined, get all merchant ids
        all_merchant_ids = get_all_merchant_ids()
        user_rated_places = get_user_rated_places(user_id)
        # option to include rated place in recommendation
        if include_rated:
            recommendation_candidate_places_ids = list(set(all_merchant_ids))
        else:
            # get user rated places
            recommendation_candidate_places_ids = list(set(all_merchant_ids) - set(user_rated_places['place_id'].tolist()))


    if len(user_rated_places) == 0:

        if not user_latitude and not user_longitude:
            # if user not rated any places and user location not defined, return all places, sorted by ratings
            if len(recommendation_candidate_places_ids) > max_returns:
                recommendation_candidate_places_ids = recommendation_candidate_places_ids[:max_returns]
            dataset = get_merchant_details(recommendation_candidate_places_ids)
            dataset.sort_values(by=['ratings'], ascending=False, inplace=True)

            return dataset.to_json(orient='records')
        elif user_latitude and user_longitude:
            # if user not rated any places and user location defined, return nearest places, sorted by ratings
            if len(nearest_places) > max_returns:
                nearest_places = nearest_places[:max_returns]
            nearest_places.sort_values(by=['ratings'], ascending=False, inplace=True)
            return nearest_places.to_json(orient='records')

    # get place factors, result from matrix factorization
    place_factors = get_place_factor_df(recommendation_candidate_places_ids)
    place_id_to_int_map = {uid: iid for iid, uid in enumerate(place_factors.columns)}
    int_to_place_id_map = {iid: uid for iid, uid in enumerate(place_factors.columns)}
    place_factors_matrix = np.array(place_factors.iloc[0].tolist()).T

    place_bias = get_place_bias(recommendation_candidate_places_ids)

    # get user factors, result from matrix factorization
    user_factors = get_user_factor_df(user_id)
    user_factors_array = np.array(user_factors[user_id].tolist())

    # get dot product of user factors and place factors, the result is predicted rating (but not with bias) given user_id
    # rating for each place
    dot_product_res = np.dot(user_factors_array, place_factors_matrix)
    dot_product_res = dot_product_res.reshape(-1, ).tolist()

    # get user bias
    user_bias = get_user_bias(user_id)
    # get global bias
    global_bias = get_global_bias()

    predictions = {}
    # we loop through all places and calculate predicted rating for each place, with bias
    for place_id in recommendation_candidate_places_ids:
        try:
            predictions[place_id] = global_bias + user_bias[user_id] + place_bias[place_id] + dot_product_res[place_id_to_int_map[place_id]]
        except KeyError:
            pass

    # list of predicted ratings, sorted by ratings
    pred_df = pd.DataFrame(list(predictions.items()), columns=['place_id', 'rating'])

    # Filter maximum number of returned places
    if len(pred_df) >= max_returns:
        # if recommendations exceed number of maximum returned places, return only maximum number of places
        top_ratings_df = pred_df.sort_values(by='rating', ascending=False).head(max_returns)
    else:
        top_ratings_df = pred_df.sort_values(by='rating', ascending=False)

    # get recommended merchant ids
    recommended_place_ids = top_ratings_df['place_id'].tolist()

    if user_latitude and user_longitude:
        dataset = nearest_places[nearest_places['_id'].isin(recommended_place_ids)]
    else:
        # get detail of recommended places, by passing merchant ids
        dataset = get_merchant_details(recommended_place_ids)
    dataset['predicted_rating'] = dataset['_id'].map(dict(zip(top_ratings_df['place_id'], top_ratings_df['rating'])))

    # Sort recommendation by predicted rating
    dataset.sort_values(by=['predicted_rating'], ascending=False, inplace=True)
    return dataset.to_json(orient='records')

