'''
    To debug, setup an env, and then run python3 app.py runserver -d with debug=True added to the app.run
'''
import pandas as pd
from flask import Flask, jsonify, request
import pickle

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
        show = input_df[input_df['id'] == predict_idx].drop(['id', 'name', 'show_rating'],axis=1).to_numpy().reshape(1, -1)

        similar_shows = model.kneighbors(
                                            X=show,
                                            n_neighbors=n_neighbors,
                                            return_distance=False
                                        )

        results = {}
        
        for show_idx in similar_shows[0]:
            if predict_idx != show_idx:
                results[str(input_df.iloc[[show_idx]]['id'].values[0])] = input_df.iloc[[show_idx]]['name'].values[0]

    except KeyError:        
        results = {}
        df_trend = input_df[input_df['show_rating']>=8.5].sample(n=n_neighbors)

        for trend in df_trend.iterrows():
            results[trend[0]] = trend[1]['name']

    return jsonify(recommendations=results)


if __name__ == '__main__':
    app.run(port = 5000)