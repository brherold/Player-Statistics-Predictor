
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



#df = pd.read_csv("DataCSVS/cleanedPlayerData41-42-43.csv", encoding='latin1')
df_total = pd.read_csv("DataCSVS/2044-Cleaned.csv", encoding='latin1')
df_PG = pd.read_csv("DataCSVS/PG-Cleaned.csv", encoding='latin1')
df_SG = pd.read_csv("DataCSVS/SG-Cleaned.csv", encoding='latin1')
df_SF = pd.read_csv("DataCSVS/SF-Cleaned.csv", encoding='latin1')
df_Perimeter = pd.read_csv("DataCSVS/Perimeter-Cleaned.csv", encoding='latin1')
df_Bigs = pd.read_csv("DataCSVS/Bigs-Cleaned.csv", encoding='latin1')


def finishing(df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ',  'Str', 'Spd', 'Sta', 'Hnd', 'Drv',
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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/FIN.pkl')


def insideShot(position, df):

    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Fin',
        'Reb', 'IQ', 'Str', 'Spd', 'Sta',
            'IS_P']

    if position == "Bigs":

        df = df[df['IS_A'] >= 3] #For Bigs
    else:
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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/IS_{position}.pkl')

    

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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/MR.pkl')

    
   
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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/3P.pkl')

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/FT.pkl')

def rebounding(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS',
        'Reb', 'IQ',  'Str', 'Spd', 'Sta',
            'Rebs']

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/RebP_{position}.pkl')

def assists(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Str', 'Spd', 'Sta',
            'AST']



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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/Ast_{position}.pkl')

def steals(position, df):
    features = ['height', 'wingspan', 'vertical', 
            'IDef', 'PDef', 'IQ', 'Str', 'Spd', 'Sta',
            'STL']

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/Stl_{position}.pkl')

def blocks(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef',  'Str', 'Spd', 'Sta',
            'BLK']

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/Blk_{position}.pkl')

#Assist to Turnover Ratio
def ast_to(position, df):
    
    df["AST/TO"] = df["AST"] / df["TO"]

    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Str', 'Spd', 'Sta',
            'AST/TO']


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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/AST-TO_{position}.pkl')

def twoPointOFG(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical',
        'IDef', 'PDef', 'IQ',  'Str', 'Spd', 'Sta',
            'O_2P_P']


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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/2OF%_{position}.pkl')




def threePointOFG(df):

    features = ['height', 'wingspan', 'vertical',
            'PDef', 'IQ','Spd', 'Sta',
            'O_3P_P']

    df = df[df['O_3P_A'] >= 4] #Use Perimeter (for both)

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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/3OF%.pkl')

def foulsDrawn(position, df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Str', 'Sta',
            'FD']

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/FD_{position}.pkl')

def OBPM(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Str', 'Sta',
            'OBPM']

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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/OBPM_{position}.pkl')



def DBPM(position,df):

    features = ['height', 'weight', 'wingspan', 'vertical',
        'Reb', 'IDef', 'PDef', 'IQ', 'Spd', 'Str', 'Sta',
            'DBPM']

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
    joblib.dump((scaler, pca, model, X_train.columns,avg_pred_value), f'Models-NoD/DBPM_{position}.pkl')


def BPM(position,df):
    features = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin',
        'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Spd', 'Str', 'Sta',
            'BPM']

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
    joblib.dump((scaler, pca, model, X_train.columns, avg_pred_value), f'Models-NoD/BPM_{position}.pkl')


#Model Training

#'''


finishing(df_total)
midRange(df_total)
threePointShooting(df_Perimeter)
freeThrowShooting(df_total)
threePointOFG(df_Perimeter)

for position in ["Perimeter","Bigs"]:
    df = pd.read_csv(f"DataCSVS/{position}-Cleaned.csv")
    insideShot(position,df)
    blocks(position,df)
    twoPointOFG(position,df)




for position in ["G","SF","Bigs"]:
    df = pd.read_csv(f"DataCSVS/{position}-Cleaned.csv")
    OBPM(position,df)
    DBPM(position,df)
    BPM(position,df)



for position in ["PG","SG","SF","Bigs"]:
    df = pd.read_csv(f"DataCSVS/{position}-Cleaned.csv")
    rebounding(position,df)
    assists(position,df)
    steals(position,df)
    ast_to(position,df)
    foulsDrawn(position,df)

#'''










#print(givePlayerStats())


