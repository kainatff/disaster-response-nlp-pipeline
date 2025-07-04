import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import nltk 
nltk.download('punkt')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    
    tokens = tokenizer.tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# loading data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_relief', engine)

#loading model
model = joblib.load("../models/classifier.pkl").set_params(n_jobs=1)

# indexing webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # computing variables required to be displayed on index page
    # extracting data needed for visuals
    # counting of messages in all the genres - direct, news, social
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    direct_counts = (df[df['genre'] == "direct"]).drop(columns=['id', 'original', 'genre', 'message']).astype(int).sum()
    direct_names = list(direct_counts.index)
    news_counts = (df[df['genre'] == "news"]).drop(columns=['id', 'original', 'genre', 'message']).astype(int).sum()
    news_names = list(news_counts.index)
    social_counts = (df[df['genre'] == "social"]).drop(columns=['id', 'original', 'genre', 'message']).astype(int).sum()
    social_names = list(social_counts.index)
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_names,
                    y=direct_counts
                )
            ],
            'layout': {
                'title': 'Distribution of category types in Direct Messages received',
                'yaxis':{
                    'title': "Count"
                }, 
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_names,
                    y=news_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category types in News Messeges',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encoding plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # rendering web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # computing variables required to be displayed on go/query page
    
    # saving user input in query
    query = request.args.get('query', '')
    
    # using model to predict classification for query
    classification_labels = model.predict([query])[0]
    print("classification label: ", classification_labels)
    
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print("classification results", classification_results)
    
    return render_template(
    'go.html',
    query=query,
    classification_result=classification_results,
    graphJSON=json.dumps([]),
    ids=[]
)
    
    
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    
if __name__ == '__main__':
    main()