import pandas as pd
from pygetPlayerSkills import get_player_info

import joblib




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
    
    return prediction[0]



scaler, pca, model, expected_columns = joblib.load(f'Models-NoD/FIN.pkl')

#print(scaler)
#print(joblib.load(f'Models-NoD/FIN.pkl')[0])

def load_model_components(filename):
    return joblib.load(f'Models-NoD/{filename}.pkl')

def givePlayerStats():
    position = input("PlayerType, Perimeter(P) or Big(B): ").lower()
    
    if "p" in position:
        position = "Perimeter"
    elif "b" in position:
        position = "Big"
    
    playerLink = input("Player's link: ")
    player = get_player_info(playerLink)
    print()
    fin_scaler, fin_pca, fin_model, fin_expected_columns = joblib.load(f'Models-NoD/FIN.pkl')


    # Load models and their components
    fin_scaler, fin_pca, fin_model, fin_expected_columns = load_model_components("FIN")
    is_scaler, is_pca, is_model, is_expected_columns = load_model_components(f"IS_{position}")
    mr_scaler, mr_pca, mr_model, mr_expected_columns = load_model_components("MR")
    tp_scaler, tp_pca, tp_model, tp_expected_columns = load_model_components("3P")
    ft_scaler, ft_pca, ft_model, ft_expected_columns = load_model_components("FT")
    rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns = load_model_components(f"RebP_{position}")
    ast_scaler, ast_pca, ast_model, ast_expected_columns = load_model_components(f"Ast_{position}")
    stl_scaler, stl_pca, stl_model, stl_expected_columns = load_model_components(f"Stl_{position}")
    blk_scaler, blk_pca, blk_model, blk_expected_columns = load_model_components(f"Blk_{position}")
    twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns = load_model_components(f"2OF%_{position}")
    threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns = load_model_components(f"3OF%_{position}")
    fd_scaler, fd_pca, fd_model, fd_expected_columns = load_model_components(f"FD_{position}")

    predicted_player_stats = {
        "Finishing%": format(preprocess_and_predict(player, fin_scaler, fin_pca, fin_model, fin_expected_columns) * 100, ".1f"),
        "InsideShot%": format(preprocess_and_predict(player, is_scaler, is_pca, is_model, is_expected_columns) * 100, ".1f"),
        "MidRange%": format(preprocess_and_predict(player, mr_scaler, mr_pca, mr_model, mr_expected_columns) * 100, ".1f"),
        "ThreePointShooting%": format(preprocess_and_predict(player, tp_scaler, tp_pca, tp_model, tp_expected_columns) * 100, ".1f"),
        "FreeThrowShooting%": format(preprocess_and_predict(player, ft_scaler, ft_pca, ft_model, ft_expected_columns) * 100, ".1f"),
        "Rebounds/G": format(preprocess_and_predict(player, rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns), ".1f"),
        "Assists/G": format(preprocess_and_predict(player, ast_scaler, ast_pca, ast_model, ast_expected_columns), ".1f"),
        "Steals/G": format(preprocess_and_predict(player, stl_scaler, stl_pca, stl_model, stl_expected_columns), ".1f"),
        "Blocks/G": format(preprocess_and_predict(player, blk_scaler, blk_pca, blk_model, blk_expected_columns), ".1f"),
        "TwoPointOFG%": format(preprocess_and_predict(player, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns) * 100, ".1f"),
        "ThreePointOFG%": format(preprocess_and_predict(player, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns) * 100, ".1f"),
        "FoulsDrawn/G": format(preprocess_and_predict(player, fd_scaler, fd_pca, fd_model, fd_expected_columns), ".1f")
    }

    return predicted_player_stats


if __name__ == "__main__":
    while(True):
        name = input("Player Name: ")
        print(givePlayerStats())
        print()
