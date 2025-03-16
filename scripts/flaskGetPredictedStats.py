import pandas as pd
from scripts.pygetPlayerSkills import get_player_info

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


    if prediction[0] < 0:
        prediction[0] = 0
    


    return prediction[0]





def load_model_components(filename):
    return joblib.load(f'Models-NoD/{filename}.pkl')

def givePlayerStats(playerLink,position):
    
    position = position.lower()

    if "p" in position:
        position = "Perimeter"
    elif "b" in position:
        position = "Big"
    
    player = get_player_info(playerLink)
    player_name = player["Name"]

    del player["Name"]

    



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
    ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns = load_model_components(f"AST-TO")


    predicted_player_stats = {
        "Finishing%": format(preprocess_and_predict(player, fin_scaler, fin_pca, fin_model, fin_expected_columns) * 100, ".1f"),
        "InsideShot%": format(preprocess_and_predict(player, is_scaler, is_pca, is_model, is_expected_columns) * 100, ".1f"),
        "MidRange%": format(preprocess_and_predict(player, mr_scaler, mr_pca, mr_model, mr_expected_columns) * 100, ".1f"),
        "3P%": format(preprocess_and_predict(player, tp_scaler, tp_pca, tp_model, tp_expected_columns) * 100, ".1f"),
        "FT%": format(preprocess_and_predict(player, ft_scaler, ft_pca, ft_model, ft_expected_columns) * 100, ".1f"),
        "Reb/G": format(preprocess_and_predict(player, rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns), ".1f"),
        "Ast/G": format(preprocess_and_predict(player, ast_scaler, ast_pca, ast_model, ast_expected_columns), ".1f"),
        "Stl/G": format(preprocess_and_predict(player, stl_scaler, stl_pca, stl_model, stl_expected_columns), ".1f"),
        "Blk/G": format(preprocess_and_predict(player, blk_scaler, blk_pca, blk_model, blk_expected_columns), ".1f"),
        "FD/G": format(preprocess_and_predict(player, fd_scaler, fd_pca, fd_model, fd_expected_columns), ".1f"),
        
        "Ast/TO": format(preprocess_and_predict(player, ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns), ".1f"),
        
        "O2P%": format(preprocess_and_predict(player, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns) * 100, ".1f"),
        "O3P%": format(preprocess_and_predict(player, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns) * 100, ".1f"),
        
    }

    return player_name, predicted_player_stats



#print(givePlayerStats("http://onlinecollegebasketball.org/prospect/205173","B"))