from db_connections import get_user_rated_places, get_all_merchant_ids, get_place_factor_df, get_user_factor_df, \
    get_user_bias, get_place_bias, get_global_bias, get_merchant_details
from nearest_recommendation import nearest_recommendation
import numpy as np
import pandas as pd


def for_you_recommendation(user_id, max_returns=10000, user_latitude=None, user_longitude=None, include_rated=False):
    if user_latitude and user_longitude:
        nearest_places = eval(nearest_recommendation(user_latitude, user_longitude))
        recommendation_candidate_places_ids = [place['_id'] for place in nearest_places]
        user_rated_places = get_user_rated_places(user_id)
        if not include_rated:
            recommendation_candidate_places_ids = list(set(recommendation_candidate_places_ids) - set(user_rated_places['place_id'].tolist()))
    else:

        all_merchant_ids = get_all_merchant_ids()
        user_rated_places = get_user_rated_places(user_id)
        if include_rated:
            recommendation_candidate_places_ids = list(set(all_merchant_ids))
        else:
            # get user rated places
            recommendation_candidate_places_ids = list(set(all_merchant_ids) - set(user_rated_places['place_id'].tolist()))


    if len(user_rated_places) == 0:
        recommendation_candidate_places_ids = recommendation_candidate_places_ids[:max_returns]
        dataset = get_merchant_details(recommendation_candidate_places_ids)
        return dataset.to_json(orient='records')

    place_factors = get_place_factor_df(recommendation_candidate_places_ids)
    place_id_to_int_map = {uid: iid for iid, uid in enumerate(place_factors.columns)}
    int_to_place_id_map = {iid: uid for iid, uid in enumerate(place_factors.columns)}
    place_factors_matrix = np.array(place_factors.iloc[0].tolist()).T

    place_bias = get_place_bias(recommendation_candidate_places_ids)

    user_factors = get_user_factor_df(user_id)
    user_factors_array = np.array(user_factors[user_id].tolist())

    dot_product_res = np.dot(user_factors_array, place_factors_matrix)
    dot_product_res = dot_product_res.reshape(-1, ).tolist()

    user_bias = get_user_bias(user_id)

    global_bias = get_global_bias()

    predictions = {}
    for place_id in recommendation_candidate_places_ids:
        try:
            predictions[place_id] = global_bias + user_bias[user_id] + place_bias[place_id] + dot_product_res[place_id_to_int_map[place_id]]
        except KeyError:
            pass

    pred_df = pd.DataFrame(list(predictions.items()), columns=['place_id', 'rating'])
    if len(pred_df) >= max_returns:
        top_ratings_df = pred_df.sort_values(by='rating', ascending=False).head(max_returns)
    else:
        top_ratings_df = pred_df.sort_values(by='rating', ascending=False)

    recommended_place_ids = top_ratings_df['place_id'].tolist()
    top_ratings_df.drop(['rating'], axis=1, inplace=True)
    dataset = get_merchant_details(recommended_place_ids)
    return dataset.to_json(orient='records')

