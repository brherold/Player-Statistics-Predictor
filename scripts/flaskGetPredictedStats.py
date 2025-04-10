import pandas as pd
from scripts.pygetPlayerSkills import get_player_info

import joblib




def preprocess_and_predict(player, scaler, pca, model, expected_columns, avg_pred=None, allow_negative=False, is_bpm =False):
    # Ensure input is in DataFrame format
    player_df = pd.DataFrame([player])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in player_df.columns:
            player_df[col] = 0

    X = player_df[expected_columns]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    prediction = model.predict(X_pca)[0]

    # If negatives aren't allowed, clip prediction at 0
    if not allow_negative:
        prediction = max(prediction, 0)
    
    comparison = "N/A"
    if avg_pred is not None:
        percent_diff = (prediction - avg_pred) / avg_pred

        if abs(percent_diff) < 0.025:  # within ±5%
            comparison = "Avg"
        elif percent_diff >= 0.025:
            comparison = "Above Avg"
        else:
            comparison = "Below Avg"
    
    if is_bpm:
        prediction = float(format(prediction, ".1f"))
        if prediction == 0.0:
            comparison = "Avg"
        elif prediction > 0.0:
            comparison = "Above Avg"
        else:
            comparison = "Below Avg"


    return {
        "prediction": prediction,
        "comparison": comparison
    }





def load_model_components(filename):
    return joblib.load(f'Models-NoD/{filename}.pkl')

def format_stat(pred, percent=False):
    value = pred["prediction"] * 100 if percent else pred["prediction"]
    return {
        "value": float(format(value, ".1f")),
        "comparison": pred["comparison"]
    }

def givePlayerStats(playerLink,position):
    
    position = position.upper()

    if position in ["PF", "C"]:
        position = "Bigs"
        position_group = "Bigs"
        

    elif position in ["PG", "SG"]:
        position_group = "Perimeter"
        

    elif position == "SF":
        position_group = "Perimeter"
        
    
    player = get_player_info(playerLink)
    player_name = player["Name"]

    del player["Name"]


    



    # Load models and their components
    fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred= load_model_components("FIN")
    is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred = load_model_components(f"IS_{position_group}")
    mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred = load_model_components("MR")
    tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred = load_model_components("3P")
    ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred = load_model_components("FT")
    rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns, rebp_avg_pred = load_model_components(f"RebP_{position}")
    ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred = load_model_components(f"Ast_{position}")
    stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred = load_model_components(f"Stl_{position}")
    blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred = load_model_components(f"Blk_{position_group}")
    twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred = load_model_components(f"2OF%_{position_group}")
    threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred = load_model_components(f"3OF%")
    fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred = load_model_components(f"FD_{position}")
    ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns, ast_to_avg_pred = load_model_components(f"AST-TO_{position}")

    obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns, obpm_avg_pred = load_model_components(f"OBPM_{position}")
    dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns, dbpm_avg_pred = load_model_components(f"DBPM_{position}")
    #bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns, bpm_avg_pred = load_model_components(f"BPM_{positionBPM}")


    # Dictionary to store all predicted stats
    predicted_player_stats = {}

    # Predict and store stats
    predicted_player_stats["Finishing%"] = format_stat(
        preprocess_and_predict(player, fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred),
        percent=True
    )

    predicted_player_stats["InsideShot%"] = format_stat(
        preprocess_and_predict(player, is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred),
        percent=True
    )

    predicted_player_stats["MidRange%"] = format_stat(
        preprocess_and_predict(player, mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred),
        percent=True
    )

    predicted_player_stats["3P%"] = format_stat(
        preprocess_and_predict(player, tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred),
        percent=True
    )

    predicted_player_stats["FT%"] = format_stat(
        preprocess_and_predict(player, ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred),
        percent=True
    )

    predicted_player_stats["Reb/G"] = format_stat(
        preprocess_and_predict(player, rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns, rebp_avg_pred)
    )

    predicted_player_stats["Ast/G"] = format_stat(
        preprocess_and_predict(player, ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred)
    )

    predicted_player_stats["Stl/G"] = format_stat(
        preprocess_and_predict(player, stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred)
    )

    predicted_player_stats["Blk/G"] = format_stat(
        preprocess_and_predict(player, blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred)
    )

    predicted_player_stats["FD/G"] = format_stat(
        preprocess_and_predict(player, fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred)
    )

    predicted_player_stats["Ast/TO"] = format_stat(
        preprocess_and_predict(player, ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns, ast_to_avg_pred)
    )

    predicted_player_stats["O2P%"] = format_stat(
        preprocess_and_predict(player, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred),
        percent=True
    )

    predicted_player_stats["O3P%"] = format_stat(
        preprocess_and_predict(player, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred),
        percent=True
    )

    predicted_player_stats["OBPM"] = format_stat(
        preprocess_and_predict(player, obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns, obpm_avg_pred, allow_negative=True, is_bpm=True)
    )

    predicted_player_stats["DBPM"] = format_stat(
        preprocess_and_predict(player, dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns, dbpm_avg_pred, allow_negative=True, is_bpm=True)
    )

    
    predicted_player_stats["BPM"] = format_stat(
        {
            "prediction": predicted_player_stats["OBPM"]["value"] + predicted_player_stats["DBPM"]["value"],
            "comparison": "Above Avg" if (predicted_player_stats["OBPM"]["value"] + predicted_player_stats["DBPM"]["value"]) > 0.0 else "Below Avg"
        }
    )

    
    return player_name, predicted_player_stats

#print(givePlayerStats("http://onlinecollegebasketball.org/prospect/202447","PG"))

            

#run python -m scripts.flaskGetPredictedStats


