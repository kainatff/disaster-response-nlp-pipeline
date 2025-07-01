from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import TreebankWordTokenizer
import sys
import nltk
import pickle
import pandas as pd

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_resources()


def load_data(database_filepath):
    # loading data into dataframe from database
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("disaster_relief", engine)
    category_names = df.columns.tolist()
    # select only messages received directly for greater accuracy
    # df = df[df['genre'] == 'direct']
    X = df.message.values
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, y, category_names


def tokenize(text):
    # cleaning and tokenizing data received.

    try:
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text)
    except LookupError as e:
        print("Error tokenizing. Ensure 'punkt' is downloaded.")
        raise e

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    # creating machine learning model using count vectorizer and tf-idf transfomer and using Gridsearch to
    # find the best parameters.

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'vect__ngram_range': [(1, 1)],
    'tfidf__use_idf': [True],
    'clf__estimator__n_estimators': [20]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)
    return cv

    #to train all 60 models comment out above parameters and use the one below
    # parameters = {
    # 'vect__ngram_range': ((1, 1), (1, 2)),           
    # 'tfidf__use_idf': (True, False),                  
    # 'clf__estimator__n_estimators': [20, 50, 100]    
    # }

def evaluate_model(model, X_test, Y_test, category_names):
    # finding model accuracy, f1score, precision,

    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('---------------------------------------------------------------------------')
        print("Category:", category_names[i])
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print("Accuracy: ", accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i]))
    print("Best Parameters:", model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Model Evaluated!')



    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()