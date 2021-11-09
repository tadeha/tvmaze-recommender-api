# import pandas as pd
from flask import Flask, jsonify, request
import pickle

# model = pickle.load(open('model.pkl','rb'))
# df_filtered = pd.read_csv('series_data.csv')
# names_df = pd.read_csv('names.csv')

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():

    '''
        Input Data:
        {
            "show_id": 512,
            "num_of_recs": 10
        }
    '''
    # data = request.get_json(force=True)
    # predict_idx = 512
    # n_neighbors = 10

    # similar_shows = model.kneighbors(X=df_filtered.iloc[predict_idx].to_numpy().reshape(1, -1),
    #                                     n_neighbors=n_neighbors+1, 
    #                                     return_distance=False
    #                                 )

    # results = []

    # for show_idx in similar_shows[0]:
    #     if predict_idx != show_idx:
    #         results.append(names_df.at[show_idx,'id'])

    # output = {'similar_shows': 'test'}

    return jsonify(results=0)

if __name__ == '__main__':
    app.run(port = 5000)