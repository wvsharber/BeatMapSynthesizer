import pandas as pd
import numpy as np
import pickle
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def load_data(difficulty):
    """Loads DataFrames saved in pickle files based on their difficulty level."""
    filelist = [f for f in os.listdir('level_df')]
    full = pd.DataFrame()
    for f in filelist:
        if f.endswith(f"{difficulty}.pkl"):
            with open(f"./level_df/{f}", 'rb') as d:
                df = pickle.load(d)
            full = pd.concat([full, df], axis = 0, ignore_index = True, sort = True)
        else:
            continue
    full.dropna(subset = list(full.iloc[:, 0:13].columns), axis = 0, inplace = True)
    full.fillna(999, inplace = True)
    return full

def train_model(model, df):
    """This function trains the model specified in 'model', either a sklearn multilabel classifier implemented
    with a Random Forest classifier, or a Chain Classifier implemented with a Random Forest classifier.
    model = 'multi' or 'chain'
    """
    X = df.iloc[:, 0: 13]
    y = df[list(filter(lambda x: str(x).startswith('notes'), df.columns))]
    
    if model == 'multi':
        multi = MultiOutputClassifier(RandomForestClassifier()).fit(X, y)
        return (multi, list(y.columns))
    elif model == 'chain':
        columns = {}
        for index, value in enumerate(y.columns):
            columns.update({value: index})
        constant = ['notes_type_0', 'notes_lineIndex_0', 'notes_lineLayer_0',
                    'notes_cutDirection_0', 'notes_type_1', 'notes_lineIndex_1', 'notes_lineLayer_1', 
                    'notes_cutDirection_1', 'notes_type_3', 'notes_lineIndex_3',
                    'notes_lineLayer_3', 'notes_cutDirection_3']
        order = [columns[x] for x in constant]
        chain = ClassifierChain(RandomForestClassifier(), order = order).fit(X, y)
        return (chain, constant)

def load_and_train():
    """This function loads the data from pickle files and trains models, saving them as pickle files in the 
    models directory."""
    difficulties = ['expert', 'expertPlus']
    models = ['chain']
    for difficulty in difficulties:
        df = load_data(difficulty)       
        print(f"Loaded {difficulty} data successfully.")
        for model in models:
            print(f"Training {difficulty} {model} model.")
            RF = train_model(model, df)
            with open(f"./models/{model}_{difficulty}.pkl", 'wb') as f:
                pickle.dump(RF, f)
            print(f"Succesfully trained and saved {difficulty} {model} model.")