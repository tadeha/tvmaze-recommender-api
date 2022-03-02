'''
    To debug, setup an env, and then run python3 app.py runserver -d with debug=True added to the app.run
    curl -X POST -H "Content-Type: application/json" \
    -d '{"show_id": 5555,"num_of_recs": 10,"weighted_model": false}' \
    http://127.0.0.1:5000/recommend
'''
from multiprocessing.sharedctypes import Value
import pandas as pd
from flask import Flask, jsonify, request
import pickle
import scipy


# Constants
MIN_YEAR = 2020
MIN_WEIGHT = 85
RANDOM_STATE = 42
COLUMNS_TO_REMOVE = ['id', 'name', 'show_rating']

# Variables
model = pickle.load(open('knn_model.pkl','rb'))
input_df = pd.read_csv('series_data.csv')

# app
app = Flask(__name__)

# routes
@app.route('/recommend', methods=['POST'])

def recommend():

    data = request.get_json(force=True)
    predict_idx = data['show_id']
    n_neighbors = data['num_of_recs']
    weighted = data['weighted_model']

    try:
        """
        show = input_df[input_df['id'] == predict_idx].drop(COLUMNS_TO_REMOVE,axis=1).to_numpy().reshape(1, -1)

        similar_shows = model.kneighbors(
                                            X=show,
                                            n_neighbors=n_neighbors,
                                            return_distance=False
                                        )

        results = {}
        
        for show_idx in similar_shows[0]:
            show_id = input_df.iloc[[show_idx]]['id'].values[0]
            if predict_idx != show_id:
                results[str(show_id)] = input_df.iloc[[show_idx]]['name'].values[0]
        """
        results = {}
        df_train = input_df.copy()
        show = df_train[df_train['id'] == predict_idx].drop(['name', 'show_rating', 'weight'], axis=1).set_index('id')
        df_train = df_train.set_index('id')
        df_train_without_metadata = df_train.drop(['name', 'show_rating', 'weight'], axis=1)
        ary = scipy.spatial.distance.cdist(show, df_train_without_metadata, metric='euclidean')
        similars = df_train[ary[0]==ary[0].min()]
        similars = similars[~similars.index.isin([predict_idx])]
        similars = similars.sample(n=n_neighbors, weights=similars.weight, random_state=RANDOM_STATE).reset_index().sort_values('weight', ascending=False)

        for row in similars.iterrows():
            show_id = row[1]['id']
            show_name = row[1]['name']
            results[str(show_id)] = show_name

    # except ValueError:
    except ValueError:
        results = {}
        df_train = input_df.copy()
        show = df_train[df_train['id'] == predict_idx].drop(['name', 'show_rating', 'weight'], axis=1).set_index('id')
        df_train = df_train.set_index('id')
        df_train_without_metadata = df_train.drop(['name', 'show_rating', 'weight'], axis=1)
        ary = scipy.spatial.distance.cdist(show, df_train_without_metadata, metric='yule')
        similars = df_train[ary[0]==ary[0].min()]
        similars = similars[~similars.index.isin([predict_idx])]
        similars = similars.sample(n=n_neighbors, weights=similars.weight, random_state=RANDOM_STATE).reset_index().sort_values('weight', ascending=False)

        for row in similars.iterrows():
            show_id = row[1]['id']
            show_name = row[1]['name']
            results[str(show_id)] = show_name

    except IndexError:        
        results = {}
        df_trend = input_df[input_df['weight']>=MIN_WEIGHT].sample(n=n_neighbors)

        for trend in df_trend.iterrows():
            results[str(trend[1]['id'])] = trend[1]['name']

    return jsonify(recommendations=results)


if __name__ == '__main__':
    app.run(port = 5000)