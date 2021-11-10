import pandas as pd
from flask import Flask, jsonify, request
import pickle
import json

model = pickle.load(open('model.pkl','rb'))
df_filtered = pd.read_csv('series_data.csv')
names_df = pd.read_csv('names.csv')

# app
app = Flask(__name__)

# routes
@app.route('/recommend', methods=['POST'])

def recommend():

    data = request.get_json(force=True)
    predict_idx = data['show_id']
    n_neighbors = data['num_of_recs']

    try:
        results = {}
        similar_shows = model.kneighbors(X=df_filtered.loc[[predict_idx]].to_numpy().reshape(1, -1),
                                            n_neighbors=n_neighbors+1,
                                            return_distance=False
                                        )

        for show_idx in similar_shows[0]:
            if predict_idx != show_idx:
                results[names_df.iloc[[show_idx]].index.tolist()[0]] = names_df.iloc[[show_idx]].values[0][0]

    except KeyError:
        results = {}
        df_trend = names_df[names_df['show_rating']>=8.5].sample(n=n_neighbors)

        for trend in df_trend.iterrows():
            results[trend[0]] = trend[1]['name']

    return jsonify(similar_show=json.dumps(dict))

if __name__ == '__main__':
    app.run(port = 5000)