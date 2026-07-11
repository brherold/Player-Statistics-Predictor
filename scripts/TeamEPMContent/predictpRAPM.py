import os
import joblib
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, module="sklearn")

'''
# Get model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "..", "pRAPM_Model", "pRAPM.pkl")


def predict_pRAPM(EPM, player_PLUS_per_Poss, team_PLUS_per_Poss):

    # Load model
    X_scaler, y_scaler, model, features, avg_pred_value, _ = joblib.load(
        model_path
    )

    # Create input dataframe
    input_data = pd.DataFrame([{
        "EPM": EPM,
        "player_PLUS_per_Poss": player_PLUS_per_Poss,
        "team_PLUS_per_Poss": team_PLUS_per_Poss
    }])

    # Ensure correct feature order
    input_data = input_data[features]

    # Scale input
    X_scaled = X_scaler.transform(input_data)

    # Predict standardized RAPM
    pred_scaled = model.predict(X_scaled)

    # Convert back to RAPM scale
    pRAPM = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    )[0][0]

    return round(float(pRAPM), 1)

    
# Example usage
if __name__ == "__main__":

    EPM = float(input("EPM: "))
    player_PLUS_per_Poss = float(input("Player PLUS per Poss: "))
    team_PLUS_per_Poss = float(input("Team PLUS per Poss: "))

    result = predict_pRAPM(
        EPM,
        player_PLUS_per_Poss,
        team_PLUS_per_Poss
    )

    print(f"\nPredicted pRAPM: {result}")

    


'''

# Get model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "..", "pRAPM_Model", "pRAPM-adj.pkl")


def predict_pRAPM(EPM, player_PLUS_per_Poss, team_PLUS_per_Poss, poss_pct_played):

    # Load model
    X_scaler, y_scaler, model, features, avg_adj_pRAPM, _ = joblib.load(
        model_path
    )

    # Create input dataframe
    input_data = pd.DataFrame([{
        "EPM": EPM,
        "player_PLUS_per_Poss": player_PLUS_per_Poss,
        "team_PLUS_per_Poss": team_PLUS_per_Poss
    }])

    # Ensure correct feature order
    input_data = input_data[features]

    # Scale features
    X_scaled = X_scaler.transform(input_data)

    # Predict standardized RAPM
    pred_scaled = model.predict(X_scaled)

    # Convert back to RAPM scale
    pRAPM = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    )[0][0]

    # Possession adjustment
    adjustment = min(poss_pct_played / 0.33, 1)

    adj_pRAPM = pRAPM * adjustment

    return round(float(adj_pRAPM), 1)

'''

# Example usage
if __name__ == "__main__":

    EPM = float(input("EPM: "))
    player_PLUS_per_Poss = float(input("Player PLUS per Poss: "))
    team_PLUS_per_Poss = float(input("Team PLUS per Poss: "))
    poss_pct_played = float(input("Possession Percentage Played: "))

    result = predict_pRAPM(
        EPM,
        player_PLUS_per_Poss,
        team_PLUS_per_Poss,
        poss_pct_played
    )

    print(f"\nPredicted pRAPM: {result}")

    
'''