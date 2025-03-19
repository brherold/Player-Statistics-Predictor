
#Input the player's name, position (Perimeter or Big), and URL and get their predicted stats for the season

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import joblib

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE



df = pd.read_csv("DataCSVS/cleanedPlayerData41-42-43.csv", encoding='latin1')


def finishing(df):
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
###################################
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/FIN.pkl')


def insideShot(position, df):
    #df = df[(df['IS-A'] >= 100)]
    df = df[(df['2OFA'] + df['3OFA']) >= 200]
    if position == "Perimeter":
        df = df[(df['IS-A'] >= 75)]
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
        df = df[(df['IS-A'] >= 100)]
        df = df[df['2OFA'] >= 0.7 * (df['2OFA'] + df['3OFA'])]  # For Bigs

    columns_to_drop = ['OS','Hnd','Fin','Rng','Drv','Pass','IDef','PDef','F-M', 'F-A', 'F%', 'IS-M', 'IS-A',
        'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('IS%', axis=1)
    y = data['IS%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/IS_{position}.pkl')

    

def midRange(df):
    df = df[df['MR-A'] >= 100]
    df = df[df['2OFA'] >= 100]

    columns_to_drop = ['Pass','Hnd','Drv','F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
                       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('MR%', axis=1)
    y = data['MR%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/MR.pkl')

    
   
def threePointShooting(df):
    df = df[(df['3P-A'] >= 100)]  # attempted ~ 2 3PA a game
    df = df[(df['2OFA'] + df['3OFA']) >= 200]

    columns_to_drop = ['IS','Reb','Fin','IDef','Pass','PDef','Str','F-M', 'F-A', 'F%', 'IS-M', 'IS-A',
       'IS%', 'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', 'DR-M', 'DR-A',
       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA',
       '2OF%', '3OFM', '3OFA', '3OF%', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('3P%', axis=1)
    y = data['3P%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/3P.pkl')

def freeThrowShooting(df):
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/FT.pkl')

def rebounding(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/RebP_{position}.pkl')

def assists(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/Ast_{position}.pkl')

def steals(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/Stl_{position}.pkl')

def blocks(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/Blk_{position}.pkl')

#Assist to Turnover Ratio
def ast_to(df):

    df = df[df['2OFA'] > 100]  # attempted ~ 2 3PA a game
    df["AST/TO"] = df["Ast"] / df["TO"]
    columns_to_drop = ['F-M', 'F-A', 'F%','IS-M', 'IS-A', 'IS%',
            'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
        'DR%', 'FTM', 'FTA', 'FT%', 'RebP','Ast', 'Stl', 'TO','2OFM', '2OFA', '2OF%',
            '3OFM', '3OFA', '3OF%', 'Blk', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('AST/TO', axis=1)
    y = data['AST/TO']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/AST-TO.pkl')

def twoPointOFG(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/2OF%_{position}.pkl')




def threePointOFG(position, df):
    df = df[df['2OFA'] >= 100]

    df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards

    columns_to_drop = ['F-M', 'F-A', 'F%', 'IS-M', 'IS-A', 'IS%',
                       'MR-M', 'MR-A', 'MR%', '3P-M', '3P-A', '3P%', 'DR-M', 'DR-A',
                       'DR%', 'FTM', 'FTA', 'FT%', 'RebP', 'Ast', 'Stl', 'Blk', '2OFM', '2OFA', '2OF%',
                       '3OFM', '3OFA', 'TO', 'PF', 'DQ', 'FD']

    df = df.drop(columns_to_drop, axis=1)

    data = df
    X = data.drop('3OF%', axis=1)
    y = data['3OF%']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/3OF%_{position}.pkl')

def foulsDrawn(position, df):
    df = df[df['2OFA'] >= 100]
    if position == "Perimeter":
        df = df[df['3OFA'] >= 0.4 * (df['2OFA'] + df['3OFA'])]  # For Guards
    else:
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns), f'Models-NoD/FD_{position}.pkl')

#Model Training
#ast_to(df)
'''


finishing(df)
midRange(df)
threePointShooting(df)
freeThrowShooting(df)

for position in ["Perimeter","Big"]:
    insideShot(position, df)
    rebounding(position, df)
    assists(position, df)
    steals(position, df)
    blocks(position, df)
    twoPointOFG(position, df) 
    threePointOFG(position, df)
    foulsDrawn(position, df)
'''



#finishing(df)






'''
def getStat(X_train, y_train):
    # Standardize the training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
'''
    







#print(givePlayerStats())


