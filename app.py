import pandas as pd
from flask import Flask, jsonify, request
import pickle

model = pickle.load(open('model.pkl','rb'))
df_filtered = pd.read_csv('series_data.csv')
names_df = pd.read_csv('names.csv')

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():

    data = request.get_json(force=True)
    predict_idx = data['show_id']
    n_neighbors = data['num_of_recs']

    similar_shows = model.kneighbors(X=df_filtered.iloc[predict_idx].to_numpy().reshape(1, -1),
                                        n_neighbors=n_neighbors+1,
                                        return_distance=False
                                    )

    results = []

    for show_idx in similar_shows[0]:
        if predict_idx != show_idx:
            results.append(names_df.at[show_idx,'id'])

    return jsonify({'similar_shows': results})

if __name__ == '__main__':
    app.run(port = 5000)