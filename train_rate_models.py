import sqlite3
import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

DB_PATH = r"D:\Hardwood\Hardwood API-BypassCloudflare\instance\basketball.db"
OUTPUT_DIR = "Models-NoD"
os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
query = """
SELECT 
    ps.height, ps.weight, ps.wingspan, ps.vertical,
    ps."IS", ps.OS, ps.Rng, ps.Fin, ps.Reb, ps.IDef, ps.PDef, ps.IQ, ps.Pass, ps.Hnd, ps.Drv, ps.Str, ps.Spd, ps.Sta,
    pa.PG_Min, pa.SG_Min, pa.SF_Min, pa.PF_Min, pa.C_Min,
    pa.AST_P, pa.TO_P, pa.ORB_P, pa.DRB_P, pa.STL_P, pa.BLK_P
FROM player_avg pa
JOIN players_skills ps ON pa.player_id = ps.player_id AND pa.season_id = ps.season_id
WHERE pa.game_type = 'College'
  AND pa.GP >= 30
  AND pa.Min >= 8
"""
df = pd.read_sql_query(query, conn)
conn.close()

min_cols = ['PG_Min', 'SG_Min', 'SF_Min', 'PF_Min', 'C_Min']
pos_map = {'PG_Min': 'PG', 'SG_Min': 'SG', 'SF_Min': 'SF', 'PF_Min': 'PF', 'C_Min': 'C'}
df['Primary_Position'] = df[min_cols].idxmax(axis=1).map(pos_map)

feature_cols = ['height', 'weight', 'wingspan', 'vertical', 'IS', 'OS', 'Rng', 'Fin', 'Reb', 'IDef', 'PDef', 'IQ', 'Pass', 'Hnd', 'Drv', 'Str', 'Spd', 'Sta']

targets = {
    'AST%': 'AST_P',
    'TOV%': 'TO_P',
    'DRB%': 'DRB_P',
    'STL%': 'STL_P',
    'BLK%': 'BLK_P'
}

positions = ['PG', 'SG', 'SF', 'PF', 'C']

print(f"Total training dataset size: {len(df)}")
for pos in positions:
    print(f"Position {pos} count: {(df['Primary_Position'] == pos).sum()}")

for stat_name, col_name in targets.items():
    print(f"\n==================== Training {stat_name} ({col_name}) ====================")
    for pos in positions:
        pos_df = df[df['Primary_Position'] == pos].copy()
        
        X = pos_df[feature_cols].values
        y = pos_df[col_name].values
        
        # 5-fold CV to evaluate performance
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for tr, te in kf.split(X, y):
            fold_model = LGBMRegressor(
                n_estimators=120,
                learning_rate=0.045,
                num_leaves=24,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            fold_model.fit(X[tr], y[tr])
            oof[te] = fold_model.predict(X[te])
            
        r2 = r2_score(y, oof)
        mae = mean_absolute_error(y, oof)
        
        # Train full model on all position data
        full_model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.045,
            num_leaves=24,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        full_model.fit(X, y)
        avg_pred_value = float(np.mean(oof))
        
        # Save model tuple: (None, None, full_model, feature_cols, avg_pred_value, df_reference)
        # scaler=None and pca=None tells preprocess_and_predict to pass raw features to LightGBM
        model_filename = f"{stat_name}_{pos}.pkl"
        out_path = os.path.join(OUTPUT_DIR, model_filename)
        
        joblib.dump((None, None, full_model, feature_cols, avg_pred_value, pos_df[feature_cols]), out_path)
        print(f"Saved {model_filename:<15} | 5-Fold R2: {r2:.3f} | MAE: {mae:.3f} | Avg: {avg_pred_value:.2f}")

print("\nAll LightGBM rate models trained and saved successfully!")
