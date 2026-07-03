import numpy as np
import joblib
from warnings import filterwarnings
import os


# Suppress sklearn feature name warnings
filterwarnings("ignore", category=UserWarning, module="sklearn")

#EPM+
#Have the models folder and you dont it saves the models 
#position, # of Possessions, # of Defensive Possessions
import numpy as np
import joblib

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_dir, "..", "..", "EPM_Model")


def predict_epm(pos, Poss, O_Poss, off_values, def_values):
    pos = pos.upper()

    # Load pipeline models (which already include StandardScaler + RidgeCV)
    # Load models
    model_off = joblib.load(os.path.join(model_folder, f"model_off_{pos}.pkl"))
    model_def = joblib.load(os.path.join(model_folder, f"model_def_{pos}.pkl"))

    # Load predictions
    y_off_pred = joblib.load(os.path.join(model_folder, f"y_off_pred_{pos}.pkl"))
    y_def_pred = joblib.load(os.path.join(model_folder, f"y_def_pred_{pos}.pkl"))

    if Poss == 0:
        Poss = 1
    if O_Poss == 0:
        O_Poss = 1

    # Convert to per 100 possessions
    off_per100 = [100 * (v or 0) / Poss for v in off_values]
    def_per100 = [100 * (v or 0) / O_Poss for v in def_values]

    # Drop "Off"/"Def" stat for PG/SG (if not used for those positions)
    if pos in ["PG", "SG"]:
        off_per100 = off_per100[:-1]
        def_per100 = def_per100[:-1]


    # === Predict Offensive EPM ===
    oepm = model_off.predict(np.array(off_per100).reshape(1, -1))[0]

    # === Predict Defensive EPM ===
    depm = model_def.predict(np.array(def_per100).reshape(1, -1))[0]

    # === Normalize EPM to target mean/std range ===
    y_off_mean = np.mean(y_off_pred)
    y_off_std = np.std(y_off_pred)

    y_def_mean = np.mean(y_def_pred)
    y_def_std = np.std(y_def_pred)

    target_mean, target_std = 0, 2.5

    oepm_scaled = ((oepm - y_off_mean) / y_off_std) * target_std + target_mean if y_off_std != 0 else target_mean
    depm_scaled = ((depm - y_def_mean) / y_def_std) * target_std + target_mean if y_def_std != 0 else target_mean


    epm = oepm_scaled + depm_scaled

    return float(round(oepm_scaled, 1)), float(round(depm_scaled, 1)), float(round(epm, 1))

'''
# Example usage
#id 213737
pos = "PF"
Poss = 46.3325
O_Poss = 46.3325
#off_features = ['FG_A','_2P_M','_3P_M','FT_M','AST','TO'](SF/PF/C) includes ['Off']
off_values = [6.1, 3.15, 0, 1.2, 1.2, 1.1,1.5]
#def_features = ['O_FG_A','O_2P_M','O_3P_M','STL','PF'] (SF/PF/C) includes ['Def']
def_values = [4.95, 2.85, 0.4, .9, 1.7 ,4.2]


#print(predict_epm(pos,Poss, O_Poss, off_values,def_values))
'''


