#Flask version of getpredictedStats

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE
from pygetPlayerSkills import get_player_info

#df = pd.read_csv("C:/Users/branh/Documents/Hardwood PROJECTSSSSSS/ML Hardwood/totalPlayerDataCleaned.csv", encoding='latin1')
df = pd.read_csv("C:/Users/branh/Documents/Hardwood PROJECTSSSSSS/ML Hardwood/cleanedPlayerData41-42.csv", encoding='latin1')

def preprocess_and_predict(player, scaler, pca, model, expected_columns):
    # Convert player data to DataFrame
    player_df = pd.DataFrame([player])
    
    # Check if player_df contains all required columns
    missing_columns = set(expected_columns) - set(player_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in player data: {missing_columns}")

    # Ensure the DataFrame columns are in the same order as expected
    player_df = player_df[expected_columns]

    # Apply the same scaling
    player_scaled = scaler.transform(player_df)

    # Apply PCA
    player_pca = pca.transform(player_scaled)

    # Predict using the model
    prediction = model.predict(player_pca)
    
    return prediction
'''
def getStat(X_train, X_test, y_train, y_test, playerLink):
    rfe = RFE(estimator=Ridge(), n_features_to_select=5)
    rfe.fit(X_train, y_train)

    # Selected features
    selected_features = X_train.columns[rfe.support_]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Ridge Regression is best model
    model = Ridge()
    pipeline = make_pipeline(StandardScaler(), model)
    param_grid = {'ridge__alpha': [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train_selected, y_train)
    y_pred = best_model.predict(X_test_selected)
    

    player = get_player_info(playerLink)
    new_data_df = pd.DataFrame([player])

    # Select only the features that were used for training
    new_data_selected = new_data_df[selected_features]

    # Predict using the model pipeline
    new_data_pred = best_model.predict(new_data_selected)

    # Print the predictions
    return new_data_pred[0]
'''
def getStat(X_train, X_test, y_train, y_test, playerLink):
    # Standardize the training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Make predictions on the test set for evaluation
    y_pred = model.predict(X_test_pca)

    # Get the player info
    player = get_player_info(playerLink)

    # Use preprocess_and_predict to get the prediction for the new player
    prediction = preprocess_and_predict(player, scaler, pca, model, X_train.columns)
    
    return prediction[0]



def finishing(df, playerLink):
    df = df[(df['F-A'] >= 30)]  # attempted ~ 2 3PA a game
    df = df[df['2OFA'] >= 100]

    columns_to_drop = ['F-M', 'F-A', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('F%', axis=1)
    y = data['F%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def insideShot(position, df, playerLink):
    
    
    if position == "Perimeter":
        df = df[(df['IS-A'] >= 50)]
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[(df['IS-A'] >= 100)]
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('IS%', axis=1)
    y = data['IS%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def midRange(df, playerLink):
    df = df[df['MR-A'] >= 100]
    df = df[df['2OFA'] >= 100]

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('MR%', axis=1)
    y = data['MR%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def threePointShooting(position,df, playerLink):
    if position == "Perimeter":
        df = df[(df['3P-A'] >= 100)]
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[(df['3P-A'] >= 70)]
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A',
                       'IS%', 'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('3P%', axis=1)
    y = data['3P%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def freeThrowShooting(df, playerLink):
    df = df[(df['FTA'] >= 95)]  # attempted ~ 2 3PA a game

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('FT%', axis=1)
    y = data['FT%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def rebounding(position, df, playerLink):
    
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('RebP', axis=1)
    y = data['RebP']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def assists(position, df, playerLink):
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Stl', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('Ast', axis=1)
    y = data['Ast']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def steals(position, df, playerLink):
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('Stl', axis=1)
    y = data['Stl']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def blocks(position, df, playerLink):
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('Blk', axis=1)
    y = data['Blk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def twoPointOFG(position, df, playerLink):
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('2OF%', axis=1)
    y = data['2OF%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def threePointOFG(position, df, playerLink):
    df = df[df['2OFA'] >= 100]
    df = df[df['3OFA'] >= 125]
    #df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('3OF%', axis=1)
    y = data['3OF%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def foulsDrawn(position, df, playerLink):
    if position == "Perimeter":
        df = df[df['2OFA'] >= 100]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[df['2OFA'] >= 150]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('FD', axis=1)
    y = data['FD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return float(getStat(X_train, X_test, y_train, y_test, playerLink))

def givePlayerStats(playerLink):
    #position = input("PlayerType, Perimeter or Big: ")
    position = "Perimeter"
    predicted_player_stats = {
        "Finishing%": format(finishing(df, playerLink) * 100, ".1f"),
        "InsideShot%": format(insideShot(position, df, playerLink) * 100, ".1f"),
        "MidRange%": format(midRange(df, playerLink) * 100, ".1f"),
        "ThreePointShooting%": format(threePointShooting(position,df, playerLink) * 100, ".1f"),
        "FreeThrowShooting%": format(freeThrowShooting(df, playerLink) * 100, ".1f"),
        "Rebounds/G": format(rebounding(position, df, playerLink), ".1f"),
        "Assists/G": format(assists(position, df, playerLink), ".1f"),
        "Steals/G": format(steals(position, df, playerLink), ".1f"),
        "Blocks/G": format(blocks(position, df, playerLink), ".1f"),
        "TwoPointOFG%": format(twoPointOFG(position, df, playerLink) * 100, ".1f"),
        "ThreePointOFG%": format(threePointOFG(position, df, playerLink) * 100, ".1f"),
        "FoulsDrawn/G": format(foulsDrawn(position, df, playerLink), ".1f")
    }
    return predicted_player_stats




#print(givePlayerStats("http://onlinecollegebasketball.org/player/178606"))

