import pandas as pd
from flask import Flask, jsonify, request
import pickle

model = pickle.load(open('model.pkl','rb'))
df_filtered = pd.read_csv('series_data.csv')
df_filtered = df_filtered.set_index('id')

model_weighted = pickle.load(open('model_weighted.pkl','rb'))
df_filtered_weighted = pd.read_csv('series_data_weighted.csv')
df_filtered_weighted = df_filtered_weighted.set_index('id')


names_df = pd.read_csv('names.csv')

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
        if weighted:
            similar_shows = model_weighted.kneighbors(X=df_filtered_weighted.loc[[predict_idx]].to_numpy().reshape(1, -1),
                                            n_neighbors=n_neighbors+1,
                                            return_distance=False
                                        )
        else:
            similar_shows = model.kneighbors(X=df_filtered.loc[[predict_idx]].to_numpy().reshape(1, -1),
                                            n_neighbors=n_neighbors+1,
                                            return_distance=False
                                        )
        results = {}
        
        for show_idx in similar_shows[0]:
            if predict_idx != show_idx:
                results[str(names_df.iloc[[show_idx]].index.tolist()[0])] = names_df.iloc[[show_idx]].values[0][1]

    except KeyError:
        results = {}
        df_trend = names_df[names_df['show_rating']>=8.5].sample(n=n_neighbors)

        for trend in df_trend.iterrows():
            results[trend[0]] = trend[1]['name']

    return jsonify(similar_show=results)

if __name__ == '__main__':
    app.run(port = 5000)