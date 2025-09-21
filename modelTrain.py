
#Input the player's name, position (Perimeter or Big), and URL and get their predicted stats for the season
#DO NOT USE STRENGTH AS A FEATURE

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



#df = pd.read_csv("DataCSVS/cleanedPlayerData41-42-43.csv", encoding='latin1')
df_total = pd.read_csv("DataCSVS/44-45-46-per56.csv", encoding='latin1')

#Remove Same Name Players on Same Teams
 
# Drop rows where player_id is 226586 or 226587 (Same names on same team)
df_total = df_total[~df_total['player_id'].isin([221906, 226586])]

# Drop rows where player_name contains "William" or "Anderson"a
df_total = df_total[~df_total['name'].str.contains("William|Anderson", case=False, na=False)]



def finishing(df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ',  'Spd', 'Sta', 'Hnd', 'Drv',
            'F_P']

    df = df[df['F_A'] >= .5]

    df = df[features]

    data = df
    X = data.drop('F_P', axis=1)
    y = data['F_P']
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

    # Now, let's predict on the test set
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/FIN.pkl')


def insideShot(position, df):

    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Fin',
        'Reb', 'IQ', 'Spd', 'Sta',
            'IS_P']
            

    if position == "Bigs":
        df = df[df["Primary_Position"].isin(["PF", "C"])]
        df = df[df['IS_A'] >= 3] #For Bigs
    else:
        df = df[df["Primary_Position"].isin(["PG", "SG","SF"])]
        df = df[df['IS_A'] >= 1] #For Perimeter

    df = df[features]


    data = df
    X = data.drop('IS_P', axis=1)
    y = data['IS_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    # Now, let's predict on the test set
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)


    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/IS_{position}.pkl')

    

def midRange(df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'OS', 'Rng',
        'Spd', 'Sta',
            'MR_P']

    df = df[df['MR_A'] >= 2]

    df = df[features]

    data = df
    X = data.drop('MR_P', axis=1)
    y = data['MR_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/MR.pkl')

    
   
def threePointShooting(df):
    features = ['height', 'weight', 'wingspan', 'vertical','OS', 'Rng', 
            'IQ', 'Spd', 'Sta',
            '_3P_P']

    df = df[df['_3P_A'] >= 4]

    df = df[features]

    data = df
    X = data.drop('_3P_P', axis=1)
    y = data['_3P_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/3P.pkl')

def freeThrowShooting(df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'OS', 'FT_P']

    df = df[df['FT_A'] >= 2]

    df = df[features]

    data = df
    X = data.drop('FT_P', axis=1)
    y = data['FT_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/FT.pkl')

def rebounding(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS',
        'Reb', 'IQ',  'Spd', 'Sta',
            'Rebs']

    df = df[df["Primary_Position"] == position]

    df = df[features]

    
    


    data = df
    X = data.drop('Rebs', axis=1)
    y = data['Rebs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/RebP_{position}.pkl')

def assists(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'AST']

    df = df[df["Primary_Position"] == position]

    df = df[features]

    

    data = df
    X = data.drop('AST', axis=1)
    y = data['AST']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/Ast_{position}.pkl')

def steals(position, df):
    features = ['height', 'wingspan', 'vertical', 
            'IDef', 'PDef', 'IQ', 'Spd', 'Sta',
            'STL']

    df = df[df["Primary_Position"] == position]

    df = df[features]

    


    data = df
    X = data.drop('STL', axis=1)
    y = data['STL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)
    
    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/Stl_{position}.pkl')

def blocks(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef',  'Spd', 'Sta',
            'BLK']

    
    df = df[df["Primary_Position"] == position]
    df = df[features]

    

    data = df
    X = data.drop('BLK', axis=1)
    y = data['BLK']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/Blk_{position}.pkl')

#Assist to Turnover Ratio
def ast_to(position, df):
    
    df["AST/TO"] = df["AST"] / df["TO"]

    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'AST/TO']

    df = df[df["Primary_Position"] == position]
    df = df[features]

    

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

    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/AST-TO_{position}.pkl')

def twoPointOFG(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical',
        'IDef', 'PDef', 'IQ',  'Spd', 'Sta',
            'O_2P_P']

    df = df[df["Primary_Position"] == position]
    df = df[features]



    data = df
    X = data.drop('O_2P_P', axis=1)
    y = data['O_2P_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/2OF%_{position}.pkl')




def threePointOFG(df):

    features = ['height', 'wingspan', 'vertical',
            'PDef', 'IQ','Spd', 'Sta',
            'O_3P_P']

    df = df[df['O_3P_A'] >= 4] #Use Perimeter (for both)


    df = df[df["Primary_Position"].isin(["PG", "SG", "SF"])]

    df = df[features]

    
    data = df
    X = data.drop('O_3P_P', axis=1)
    y = data['O_3P_P']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)

    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/3OF%.pkl')

def foulsDrawn(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'FD']
    
    df = df[df["Primary_Position"] == position]
    df = df[features]



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
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/FD_{position}.pkl')

def OBPM(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'OBPM']

    df = df[df["Primary_Position"] == position]
    df = df[features]



    data = df
    X = data.drop('OBPM', axis=1)
    y = data['OBPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/OBPM_{position}.pkl')



def DBPM(position,df):

    features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef', 'IQ', 'Spd', 'Sta',
            'DBPM']

    df = df[df["Primary_Position"] == position]
    df = df[features]



    data = df
    X = data.drop('DBPM', axis=1)
    y = data['DBPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'Models-NoD/DBPM_{position}.pkl')


def BPM(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'BPM']

    df = df[df["Primary_Position"] == position]
    df = df[features]

    data = df
    X = data.drop('BPM', axis=1)
    y = data['BPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Models-NoD/BPM_{position}.pkl')


#EPM+

def OEPM(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'OEPM']
    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]

    data = df
    X = data.drop('OEPM', axis=1)
    y = data['OEPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'EPM+_Model//OPM+_{position}.pkl')



def DEPM(position,df):

    features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef', 'IQ', 'Spd', 'Sta',
            'DEPM']
    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]



    data = df
    X = data.drop('DEPM', axis=1)
    y = data['DEPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value, df), f'EPM+_Model//DPM+_{position}.pkl')


def EPM_(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'EPM']
    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]



    data = df
    X = data.drop('EPM', axis=1)
    y = data['EPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'EPM+_Model/EPM+_{position}.pkl')




#Adj_EPM+

def adj_OEPM(position,df):

    if position in ["PG", "SG"]:
        features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
         'IQ', 'Pass', 'Hnd', 'Drv', 'Spd',   'Sta',
            'adj_OEPM_scaled']
    else:
        features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd',   'Sta',
            'adj_OEPM_scaled']


    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]

    feat = features[-1]


    data = df
    X = data.drop(feat, axis=1)
    y = data[feat]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Adj_EPM+_Model/Adj_OPM+_{position}.pkl')



def adj_DEPM(position,df):
    
    if position in ["PG", "SG"]:
        features = ['height', 'weight', 'wingspan', 'vertical',
         'IDef', 'PDef', 'IQ', 'Spd',  'Sta',
            'adj_DEPM_scaled']
    else:
        features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef', 'IQ', 'Spd', 'Sta',
            'adj_DEPM_scaled']
        
    

    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]

    feat = features[-1]


    data = df
    X = data.drop(feat, axis=1)
    y = data[feat]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Adj_EPM+_Model/Adj_DPM+_{position}.pkl')


def adj_EPM(position,df):
    if position in ["PG", "SG"]:
        features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
         'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'adj_EPM_scaled']
    else:
        features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Sta',
            'adj_EPM_scaled']
    
    df = df.copy()

    df = df[df["Primary_Position"] == position]

    df = df[features]

    feat = features[-1]


    data = df
    X = data.drop(feat, axis=1)
    y = data[feat]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train Ridge Regression model
    model = Ridge()
    model.fit(X_train_pca, y_train)
    
    X_test_scaled = scaler.transform(X_test)  
    X_test_pca = pca.transform(X_test_scaled)  # Transform the test set using the fitted PCA

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_pca)

    # Calculate the average of the predicted values for the test set
    avg_pred_value = np.mean(y_pred_test)

    # Save model, scaler, and PCA
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value, df), f'Adj_EPM+_Model/Adj_EPM+_{position}.pkl')
#Model Training


#'''

finishing(df_total)
midRange(df_total)
threePointShooting(df_total)
freeThrowShooting(df_total)
threePointOFG(df_total)

for position in ["Perimeter","Bigs"]:
    
    insideShot(position,df_total)
    







for position in ["PG","SG","SF","PF",'C']:
    rebounding(position,df_total)
    assists(position,df_total)
    steals(position,df_total)
    blocks(position,df_total)
    ast_to(position,df_total)
    twoPointOFG(position,df_total)
    foulsDrawn(position,df_total)
    OBPM(position,df_total)
    DBPM(position,df_total)
    BPM(position,df_total)

for position in ["PG","SG","SF","PF",'C']:

    OEPM(position,df_total)
    DEPM(position,df_total)
    EPM_(position,df_total)





for position in ["PG","SG","SF","PF",'C']:
    df = pd.read_csv(f"DataCSVS//44-45-46-AdjEPM-Scaled.csv")
    adj_OEPM(position,df)
    adj_DEPM(position,df)
    adj_EPM(position,df)



#'''








#print(givePlayerStats())


