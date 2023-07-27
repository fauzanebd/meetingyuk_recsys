from db_connections import get_all_merchants, get_all_merchant_id_and_locs, get_merchant_details
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import pickle
from kneed import KneeLocator
from sklearn.cluster import KMeans
import pandas as pd



def nearest_recommendation(latitude, longitude, max_returns=10000, from_api=False, max_radius=10000):
    """
    This function return nearby places
    :param latitude: user latitude
    :param longitude: user longitude
    :param max_returns: number of maximum places returned
    :param from_api: if True, return places with json format
    :param max_radius: maximum radius to search for nearest places
    :return:
    """
    dfcluster, kmeans = get_kmeans_model()

    places = get_all_merchant_id_and_locs()
    places.columns = ["_id", "latitude", "longitude"]
    user_loc = np.array([float(latitude), float(longitude)]).reshape(1, -1)
    cluster = kmeans.predict(user_loc)[0]
    recommendation = places[dfcluster == cluster]
    # recommendation = recommendation[[
    #     '_id', 'owner_id', 'name', 'address', 'latitude', 'longitude',
    #     'ratings', 'review_count', 'opening_hours', 'image_url',
    #     'rooms', 'tag_ids'
    # ]]
    # recommendation['latitude'] = pd.to_numeric(recommendation['latitude'], errors='coerce')
    # recommendation['longitude'] = pd.to_numeric(recommendation['longitude'], errors='coerce')

    # recommendation['distance'] = haversine_distances(
    #     recommendation[['latitude', 'longitude']], user_loc
    # )
    # recommendation['distance'] = recommendation['distance'] * 6371
    # recommendation = recommendation[recommendation['distance'] <= max_radius]

    # Convert to radians
    recommendation_rad = np.radians(recommendation[['latitude', 'longitude']])
    user_loc_rad = np.radians(user_loc)

    # Compute distances
    distances_rad = haversine_distances(recommendation_rad, user_loc_rad)
    distances_km = distances_rad * 6371.0

    # Assign distances to the recommendation dataframe
    recommendation['distance'] = distances_km[:, 0]  # Taking the first column since user_loc is a single point

    # Filter out recommendations beyond the max_radius
    recommendation = recommendation[recommendation['distance'] <= max_radius]

    # if recommendation is empty after filtered, raise error:
    if recommendation.empty:
        raise ValueError("No recommendations found within the radius of {} km".format(max_radius))

    if len(recommendation) > max_returns:
        sort = recommendation.sort_values(by=['distance'], ascending=True)[:max_returns]
    else:
        sort = recommendation.sort_values(by=['distance'], ascending=True)



    recommended_place_ids = sort['_id'].tolist()
    dataset = get_merchant_details(recommended_place_ids)
    dataset['distance_in_km'] = dataset['_id'].map(dict(zip(sort['_id'], sort['distance'])))
    dataset.sort_values(by=['distance_in_km'], inplace=True)
    if from_api:
        result = dataset.to_json(orient='records')
    else:
        result = dataset
    return result



def get_kmeans_model():
    with open("models/kmeans_model.pkl", "rb") as f:
      model = pickle.load(f)

    with open ("objects/data.obj","rb") as v:
      dfcluster = pickle.load(v)
    return dfcluster, model


def find_k():
    """
    This function find the best k for kmeans
    :return: Cluster of kmeans model
    """
    print("Start calculating kmeans")
    df = get_all_merchants()
    df['latitude'] = df['location.latitude']
    df['longitude'] = df['location.longitude']
    coords = df[['latitude', 'longitude']]
    distortions = []
    K = range(1, 21)
    for k in K:
        kmeansModel = KMeans(n_clusters=k)
        kmeansModel = kmeansModel.fit(coords)
        distortions.append(kmeansModel.inertia_)

    kn = KneeLocator(range(1, 21), distortions, curve='convex', direction='decreasing')
    k = kn.knee

    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(coords)
    df['cluster'] = kmeans.predict(df[['latitude', 'longitude']])
    dfcluster = df['cluster']

    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open("objects/data.obj", "wb") as v:
        pickle.dump(dfcluster, v)

    print("Finish calculating kmeans")
    return dfcluster, kmeans
