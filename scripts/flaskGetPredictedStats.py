import pandas as pd
from scripts.pygetPlayerSkills import get_player_info

import joblib


def percentile_to_rgb(percentile):
    # Clamp between 0 and 100
    p = max(0, min(100, percentile))

    if p <= 50:
        # Interpolate from bright red (255,50,50) to gray (70,70,75)
        t = p / 50
        r = int(255 + (70 - 255) * t)
        g = int(50 + (70 - 50) * t)
        b = int(50 + (75 - 50) * t)
    else:
        # Interpolate from gray (70,70,75) to bright green (0,200,100)
        t = (p - 50) / 50
        r = int(70 + (0 - 70) * t)
        g = int(70 + (200 - 70) * t)
        b = int(75 + (100 - 75) * t)

    return f'rgb({r},{g},{b})'



def preprocess_and_predict(df, player, scaler, pca, model, expected_columns, avg_pred=None, allow_negative=False, is_bpm =False, opposite_comparison = False):
    # Ensure input is in DataFrame format
    player_df = pd.DataFrame([player])


    
    
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in player_df.columns:
            player_df[col] = 0
        if col not in df.columns:
            df[col] = 0


        
    # 1. Predict for all players in df
    X_all = df[expected_columns].copy()

    # Scale and transform
    X_scaled_all = scaler.transform(X_all)
    X_pca_all = pca.transform(X_scaled_all)
    preds_all = model.predict(X_pca_all)
    

    X = player_df[expected_columns]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    prediction = model.predict(X_pca)[0]


    # If negatives aren't allowed, clip prediction at 0
    if not allow_negative:
        prediction = max(prediction, 0)



    
    if is_bpm:
        prediction = float(format(prediction, ".1f"))
        if prediction == 0.0:
            comparison = "Avg"
        elif prediction > 0.0:
            comparison = "Above Avg"
        else:
            comparison = "Below Avg"
        
        
    #Get Percentile
    percentile = (preds_all < prediction).mean() * 100
    if opposite_comparison == True:
        percentile = 100 - percentile

    percentile = int(round(percentile,0))



    return {
        "prediction": prediction,
        "percentile": percentile,
        "color": percentile_to_rgb(percentile)
    }





def load_model_components(filename):
    return joblib.load(f'Models-NoD/{filename}.pkl')

def format_stat(pred, percent=False):
    value = pred["prediction"] * 100 if percent else pred["prediction"]
    return {
        "value": float(format(value, ".1f")),
        "percentile": pred["percentile"],
        "color": pred["color"]
    }

def givePlayerStats(playerLink,position):
    
    position = position.upper()

    if position in ["PF", "C"]:
        position_group = "Bigs"
        

    elif position in ["PG", "SG"]:
        position_group = "Perimeter"
        

    elif position == "SF":
        position_group = "Perimeter"
        
    
    player = get_player_info(playerLink)
    player_name = player["Name"]

    del player["Name"]


    



    # Load models and their components
    fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred, df= load_model_components("FIN")
    is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred , df= load_model_components(f"IS_{position_group}")
    mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred, df = load_model_components("MR")
    tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred , df= load_model_components("3P")
    ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred , df= load_model_components("FT")
    rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns, rebp_avg_pred , df= load_model_components(f"RebP_{position}")
    ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred , df= load_model_components(f"Ast_{position}")
    stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred, df = load_model_components(f"Stl_{position}")
    blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred, df = load_model_components(f"Blk_{position}")
    twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred, df = load_model_components(f"2OF%_{position}")
    threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred, df = load_model_components(f"3OF%")
    fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred , df= load_model_components(f"FD_{position}")
    ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns, ast_to_avg_pred, df = load_model_components(f"AST-TO_{position}")

    obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns, obpm_avg_pred, df = load_model_components(f"OBPM_{position}")
    dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns, dbpm_avg_pred , df= load_model_components(f"DBPM_{position}")
    bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns, bpm_avg_pred, df = load_model_components(f"BPM_{position}")


    # Dictionary to store all predicted stats
    predicted_player_stats = {}

    # Predict and store stats
    predicted_player_stats["Fin%"] = format_stat(
        preprocess_and_predict(df, player, fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred),
        percent=True
    )

    predicted_player_stats["IS%"] = format_stat(
        preprocess_and_predict(df, player, is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred),
        percent=True
    )

    predicted_player_stats["Mid%"] = format_stat(
        preprocess_and_predict(df, player, mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred),
        percent=True
    )

    predicted_player_stats["3PT%"] = format_stat(
        preprocess_and_predict(df, player, tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred),
        percent=True
    )

    predicted_player_stats["FT%"] = format_stat(
        preprocess_and_predict(df, player, ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred),
        percent=True
    )

    predicted_player_stats["Reb"] = format_stat(
        preprocess_and_predict(df, player, rebp_scaler, rebp_pca, rebp_model, rebp_expected_columns, rebp_avg_pred)
    )

    predicted_player_stats["Ast"] = format_stat(
        preprocess_and_predict(df, player, ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred)
    )

    predicted_player_stats["Stl"] = format_stat(
        preprocess_and_predict(df, player, stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred)
    )

    predicted_player_stats["Blk"] = format_stat(
        preprocess_and_predict(df, player, blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred)
    )

    predicted_player_stats["FD"] = format_stat(
        preprocess_and_predict(df, player, fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred)
    )

    predicted_player_stats["Ast/TO"] = format_stat(
        preprocess_and_predict(df, player, ast_to_scaler, ast_to_pca, ast_to_model, ast_to_expected_columns, ast_to_avg_pred)
    )

    predicted_player_stats["O2%"] = format_stat(
        preprocess_and_predict(df, player, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred,opposite_comparison=True),
        percent=True
    )

    predicted_player_stats["O3%"] = format_stat(
        preprocess_and_predict(df, player, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred,opposite_comparison=True),
        percent=True
    )

    predicted_player_stats["OPM"] = format_stat(
        preprocess_and_predict(df, player, obpm_scaler, obpm_pca, obpm_model, obpm_expected_columns, obpm_avg_pred, allow_negative=True, is_bpm=True)
    )

    predicted_player_stats["DPM"] = format_stat(
        preprocess_and_predict(df, player, dbpm_scaler, dbpm_pca, dbpm_model, dbpm_expected_columns, dbpm_avg_pred, allow_negative=True, is_bpm=True)
    )

    #Get percentile for EPM
    epm_percentile = preprocess_and_predict(df, player, bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns, bpm_avg_pred, allow_negative=True, is_bpm=True)["percentile"]
    epm_color = preprocess_and_predict(df, player, bpm_scaler, bpm_pca, bpm_model, bpm_expected_columns, bpm_avg_pred, allow_negative=True, is_bpm=True)["color"]
    predicted_player_stats["EPM"] = format_stat(
        {
            "prediction": predicted_player_stats["OPM"]["value"] + predicted_player_stats["DPM"]["value"],
            "percentile": epm_percentile,
            "color": epm_color
        }
    )

    
    return player_name, predicted_player_stats

#print(givePlayerStats("https://onlinecollegebasketball.org/player/205173/A","SG"))

            

#run python -m scripts.flaskGetPredictedStats


