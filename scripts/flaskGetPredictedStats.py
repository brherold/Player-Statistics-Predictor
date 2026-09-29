import os
import pandas as pd
import numpy as np
from scripts.pygetPlayerSkills import flask_get_player_info

import joblib

# Fix backwards compatibility when unpickling DataFrames pickled with different pandas/StringDtype versions
try:
    from pandas.core.arrays.string_ import StringDtype
    _orig_string_dtype_init = StringDtype.__init__

    def _safe_string_dtype_init(self, *args, **kwargs):
        if len(args) > 2:
            args = args[:2]
        return _orig_string_dtype_init(self, *args, **kwargs)

    StringDtype.__init__ = _safe_string_dtype_init
except Exception:
    pass

RAPM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "positional_rapm_models.joblib")
_rapm_bundle = None

def get_rapm_bundle():
    global _rapm_bundle
    if _rapm_bundle is None:
        _rapm_bundle = joblib.load(RAPM_MODEL_PATH)
    return _rapm_bundle



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






def preprocess_and_predict(df, player_df, scaler, pca, model, expected_columns, avg_pred=None, allow_negative=False, is_bpm =False, opposite_comparison = False):
    


    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in player_df.columns:
            player_df[col] = 0
        if col not in df.columns:
            df[col] = 0


        
    # 1. Predict for all players in df
    X_all = df[expected_columns].copy()
    X = player_df[expected_columns].copy()

    # Scale and transform if scaler and pca provided, else pass raw features
    if scaler is not None:
        X_scaled_all = scaler.transform(X_all)
        X_scaled = scaler.transform(X)
    else:
        X_scaled_all = X_all.values
        X_scaled = X.values

    if pca is not None:
        X_pca_all = pca.transform(X_scaled_all)
        X_pca = pca.transform(X_scaled)
    else:
        X_pca_all = X_scaled_all
        X_pca = X_scaled

    preds_all = model.predict(X_pca_all)
    prediction = float(model.predict(X_pca)[0])


    # If negatives aren't allowed, clip prediction at 0
    if not allow_negative:
        prediction = max(prediction, 0)



    
    if is_bpm:
        prediction = float(format(prediction, ".1f"))

        
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
    if "Adj_" in filename:
        return joblib.load(f'Adj_EPM+_Model/{filename}.pkl')
    elif "+" in filename:
        return joblib.load(f'EPM+_Model/{filename}.pkl')
    else:
        return joblib.load(f'Models-NoD/{filename}.pkl')

def format_stat(pred, percent=False):
    value = pred["prediction"] * 100 if percent else pred["prediction"]
    return {
        "value": float(format(value, ".1f")),
        "percentile": pred["percentile"],
        "color": pred["color"]
    }

def givePlayerStats(player_html_link,position, from_file=False):

    position = position.upper()

    if position in ["PF", "C"]:
        position_group = "Bigs"
        

    elif position in ["PG", "SG"]:
        position_group = "Perimeter"
        

    elif position == "SF":
        position_group = "Perimeter"
    
    #print("YO")
    player , playerID = flask_get_player_info(player_html_link, from_file)
    #print(player)
    player_name = player["Name"]
    
    del player["Name"]


    # Convert Player Dictionary to DF
    player_df = pd.DataFrame([player])


    



    # Load models and their components
    fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred, fin_df = load_model_components("FIN")
    is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred, is_df = load_model_components(f"IS_{position_group}")
    mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred, mr_df = load_model_components("MR")
    tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred, tp_df = load_model_components("3P")
    ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred, ft_df = load_model_components("FT")

    # Positional Rate Stat Models
    ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred, ast_df = load_model_components(f"AST%_{position}")
    tov_scaler, tov_pca, tov_model, tov_expected_columns, tov_avg_pred, tov_df = load_model_components(f"TOV%_{position}")
    drb_scaler, drb_pca, drb_model, drb_expected_columns, drb_avg_pred, drb_df = load_model_components(f"DRB%_{position}")
    stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred, stl_df = load_model_components(f"STL%_{position}")
    blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred, blk_df = load_model_components(f"BLK%_{position}")

    twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred, twoof_df = load_model_components(f"2OF%_{position}")
    threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred, threeof_df = load_model_components("3OF%")
    fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred, fd_df = load_model_components(f"FD_{position}")

    # Dictionary to store all predicted stats
    predicted_player_stats = {}

    # Predict and store shooting stats
    predicted_player_stats["Fin%"] = format_stat(
        preprocess_and_predict(fin_df, player_df, fin_scaler, fin_pca, fin_model, fin_expected_columns, fin_avg_pred),
        percent=True
    )

    predicted_player_stats["IS%"] = format_stat(
        preprocess_and_predict(is_df, player_df, is_scaler, is_pca, is_model, is_expected_columns, is_avg_pred),
        percent=True
    )

    predicted_player_stats["Mid%"] = format_stat(
        preprocess_and_predict(mr_df, player_df, mr_scaler, mr_pca, mr_model, mr_expected_columns, mr_avg_pred),
        percent=True
    )

    predicted_player_stats["3PT%"] = format_stat(
        preprocess_and_predict(tp_df, player_df, tp_scaler, tp_pca, tp_model, tp_expected_columns, tp_avg_pred),
        percent=True
    )

    predicted_player_stats["FT%"] = format_stat(
        preprocess_and_predict(ft_df, player_df, ft_scaler, ft_pca, ft_model, ft_expected_columns, ft_avg_pred),
        percent=True
    )

    # Predict and store rate stats
    predicted_player_stats["AST%"] = format_stat(
        preprocess_and_predict(ast_df, player_df, ast_scaler, ast_pca, ast_model, ast_expected_columns, ast_avg_pred)
    )

    predicted_player_stats["TOV%"] = format_stat(
        preprocess_and_predict(tov_df, player_df, tov_scaler, tov_pca, tov_model, tov_expected_columns, tov_avg_pred, opposite_comparison=True)
    )

    predicted_player_stats["DRB%"] = format_stat(
        preprocess_and_predict(drb_df, player_df, drb_scaler, drb_pca, drb_model, drb_expected_columns, drb_avg_pred)
    )

    predicted_player_stats["STL%"] = format_stat(
        preprocess_and_predict(stl_df, player_df, stl_scaler, stl_pca, stl_model, stl_expected_columns, stl_avg_pred)
    )

    predicted_player_stats["BLK%"] = format_stat(
        preprocess_and_predict(blk_df, player_df, blk_scaler, blk_pca, blk_model, blk_expected_columns, blk_avg_pred)
    )

    predicted_player_stats["FD"] = format_stat(
        preprocess_and_predict(fd_df, player_df, fd_scaler, fd_pca, fd_model, fd_expected_columns, fd_avg_pred)
    )

    predicted_player_stats["O2%"] = format_stat(
        preprocess_and_predict(twoof_df, player_df, twoof_scaler, twoof_pca, twoof_model, twoof_expected_columns, twoof_avg_pred, opposite_comparison=True),
        percent=True
    )

    predicted_player_stats["O3%"] = format_stat(
        preprocess_and_predict(threeof_df, player_df, threeof_scaler, threeof_pca, threeof_model, threeof_expected_columns, threeof_avg_pred, opposite_comparison=True),
        percent=True
    )

    # Positional RAPM (replaces EPM+, OPM+, DPM+)
    rapm_bundle = get_rapm_bundle()
    pos_key = position.lower()
    if pos_key in rapm_bundle["models"]:
        pos_models = rapm_bundle["models"][pos_key]
        feature_sets = rapm_bundle["feature_sets"]
        ref_dists = rapm_bundle.get("reference_distributions", {}).get(pos_key, {})

        # Predict oRAPM and dRAPM
        for target in ["oRAPM", "dRAPM"]:
            model = pos_models[target]
            cols = feature_sets[target]
            for col in cols:
                if col not in player_df.columns:
                    player_df[col] = 0.0
            X = player_df[cols].values
            pred_val = float(model.predict(X)[0])
            ref_dist = ref_dists.get(target, np.array([]))
            if len(ref_dist) > 0:
                pct = float(np.clip((ref_dist < pred_val).mean() * 100.0, 0.0, 100.0))
            else:
                pct = 50.0
            pct_int = int(round(pct))
            rounded_val = float(format(pred_val, ".1f")) + 0.0

            predicted_player_stats[target] = {
                "value": rounded_val,
                "percentile": pct_int,
                "color": percentile_to_rgb(pct_int)
            }

        # Option 1: Strictly enforce RAPM = oRAPM + dRAPM for additive consistency
        total_rapm = round(predicted_player_stats["oRAPM"]["value"] + predicted_player_stats["dRAPM"]["value"], 1) + 0.0
        rapm_ref_dist = ref_dists.get("RAPM", np.array([]))
        if len(rapm_ref_dist) > 0:
            rapm_pct = float(np.clip((rapm_ref_dist < total_rapm).mean() * 100.0, 0.0, 100.0))
        else:
            rapm_pct = 50.0
        rapm_pct_int = int(round(rapm_pct))

        predicted_player_stats["RAPM"] = {
            "value": total_rapm,
            "percentile": rapm_pct_int,
            "color": percentile_to_rgb(rapm_pct_int)
        }


    
    return player_name, predicted_player_stats, playerID



#print(givePlayerStats("https://onlinecollegebasketball.org/player/205173/A","C"))

#run python -m scripts.flaskGetPredictedStats


#print(givePlayerStats("C:/Users/branh/Documents/Hardwood PROJECTSSSSSS/StatPredictor-Manual/PlayersInputted/214216-H.html","SG", True))