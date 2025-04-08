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
    
    return prediction[0]



scaler, pca, model, expected_columns = joblib.load(f'Models-NoD/FIN.pkl')



def load_model_components(filename):
    return joblib.load(f'Models-NoD/{filename}.pkl')

def givePlayerStats():
    position = input("Player Position (pg) (sg) (sf) (pf) (c):  ").upper()
    
    if position in ["PF", "C"]:
        position = "Bigs"
        position_group = "Bigs"
        positionBPM = "Bigs"

    elif position in ["PG", "SG"]:
        position_group = "Perimeter"
        positionBPM = "G"

    elif position == "SF":
        position_group = "Perimeter"
        positionBPM = "SF"
    
    
    playerLink = input("Player's link: ")
    player = get_player_info(playerLink)
    player_name = player["Name"]

    del player["Name"]
    print()
    print(player_name)
    


    # Load models and their components
    fin_scaler, fin_pca, fin_model, fin_expected_columns = load_model_components("FIN")
    is_scaler, is_pca, is_model, is_expected_columns = load_model_components(f"IS_{position_group}")
    mr_scaler, mr_pca, mr_model, mr_expected_columns = load_model_components("MR")
    tp_scaler, tp_pca, tp_model, tp_expected_columns = load_model_components("3P")
    ft_scaler, ft_pca, ft_model, ft_expected_columns = load_model_components("FT")
    rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns = load_model_components(f"RebP_{position}")
    ast_scaler, ast_pca, ast_model, ast_expected_columns = load_model_components(f"Ast_{position}")
    stl_scaler, stl_pca, stl_model, stl_expected_columns = load_model_components(f"Stl_{position}")
    blk_scaler, blk_pca, blk_model, blk_expected_columns = load_model_components(f"Blk_{position_group}")
    twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns = load_model_components(f"2OF%_{position_group}")
    threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns = load_model_components(f"3OF%")
    fd_scaler, fd_pca, fd_model, fd_expected_columns = load_model_components(f"FD_{position}")
    astTo_scaler, astTO_pca, astTO_model, astTO_expected_columns = load_model_components(f"AST-TO_{position}")

    obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns = load_model_components(f"OBPM_{positionBPM}")
    dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns = load_model_components(f"DBPM_{positionBPM}")
    bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns = load_model_components(f"BPM_{positionBPM}")


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
        "AST/TO" : format(preprocess_and_predict(player, astTo_scaler, astTO_pca, astTO_model, astTO_expected_columns), ".1f"),
        "TwoPointOFG%": format(preprocess_and_predict(player, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns) * 100, ".1f"),
        "ThreePointOFG%": format(preprocess_and_predict(player, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns) * 100, ".1f"),
        "FD/G": format(preprocess_and_predict(player, fd_scaler, fd_pca, fd_model, fd_expected_columns), ".1f"),
        "OBPM": format(preprocess_and_predict(player, obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns), ".1f"),
        "DBPM": format(preprocess_and_predict(player, dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns), ".1f"),
        "BPM": format(preprocess_and_predict(player, bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns), ".1f"),

    }

    return predicted_player_stats


if __name__ == "__main__":
    while(True):
        print(givePlayerStats())
        print()

#python -m scripts.getPredictedStats