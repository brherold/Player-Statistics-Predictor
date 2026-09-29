import os
import re
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, module="sklearn")


class HybridEnsembleModel:
    """
    Ensemble combining HistGradientBoostingRegressor (75% weight)
    and L2-Regularized Ridge Regression (25% weight) for optimal stability.
    """
    def __init__(self, gbr_params=None, ridge_alpha=15.0, gbr_weight=0.75):
        self.gbr_weight = gbr_weight
        self.ridge_weight = 1.0 - gbr_weight

    def predict(self, X):
        p_gbr = self.gbr.predict(X)
        p_ridge = self.ridge.predict(X)
        return self.gbr_weight * p_gbr + self.ridge_weight * p_ridge


class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'HybridEnsembleModel':
            return HybridEnsembleModel
        return super().find_class(module, name)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "bRAPM_Model", "brapm_models.pkl"),
    r"D:\Omp Proj\Hardwood GameLog Reader\bRAPM\models\brapm_models.pkl",
]

_CACHED_BUNDLE = None


def load_brapm_bundle(model_path=None):
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None:
        return _CACHED_BUNDLE

    path_to_try = model_path
    if not path_to_try:
        for c in MODEL_CANDIDATES:
            if os.path.exists(c):
                path_to_try = c
                break

    if not path_to_try or not os.path.exists(path_to_try):
        raise FileNotFoundError(f"bRAPM model bundle not found at {path_to_try or MODEL_CANDIDATES}")

    with open(path_to_try, "rb") as f:
        _CACHED_BUNDLE = ModelUnpickler(f).load()

    return _CACHED_BUNDLE


def parse_min_tooltip(cell) -> dict:
    tooltip = cell.get('title', '') or cell.get('alt', '')
    result = {'PG_Min': 0.0, 'SG_Min': 0.0, 'SF_Min': 0.0, 'PF_Min': 0.0, 'C_Min': 0.0}
    if not tooltip:
        return result
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        m = re.search(rf'\b{pos}\s+([\d.]+)', tooltip)
        if m:
            result[f'{pos}_Min'] = float(m.group(1))
    return result


def parse_ofg_tooltip(cell) -> dict:
    tooltip = cell.get('title', '') or cell.get('alt', '')
    result = {}
    if not tooltip:
        return result
    for line in tooltip.strip().split('\n'):
        line = line.strip()
        m_pct_ma = re.match(r'(\S+)\s+([\d.]+)%\s+\((\d+)-(\d+)\)', line)
        if m_pct_ma:
            label, pct, made, att = m_pct_ma.groups()
            result[label] = {'pct': float(pct), 'M': int(made), 'A': int(att)}
            continue
        m_pct = re.match(r'(\S+)\s+([\d.]+)%', line)
        if m_pct:
            label, pct = m_pct.groups()
            result[label] = float(pct)
            continue
        m_val = re.match(r'(\S+)\s+([\d.]+)$', line)
        if m_val:
            label, val = m_val.groups()
            result[label] = float(val)
    return result


def parse_team_html_to_df(soup_or_html):
    if isinstance(soup_or_html, str):
        soup = BeautifulSoup(soup_or_html, "html.parser")
    else:
        soup = soup_or_html

    table = soup.find('table')
    if not table:
        return pd.DataFrame(), {}, {}

    rows = table.find_all('tr')
    headers = [c.get_text(strip=True) for c in rows[0].find_all(['th', 'td'])]
    ofg_idx = headers.index('OFG') if 'OFG' in headers else None
    min_idx = headers.index('Min') if 'Min' in headers else None

    player_records = []
    team_total = {}
    opp_total = {}

    for r in rows[1:]:
        cells = r.find_all(['td', 'th'])
        if not cells:
            continue
        row_text = [c.get_text(strip=True) for c in cells]
        if not row_text or not row_text[0]:
            continue

        row_label = row_text[0]
        link = cells[0].find('a')

        if row_label.lower() == 'total':
            for h, val in zip(headers[1:], row_text[1:]):
                clean_v = val.replace('+', '').replace('%', '').strip()
                team_total[h] = pd.to_numeric(clean_v, errors='coerce')
            continue

        if row_label.lower() in ('opponents', 'opponent'):
            for h, val in zip(headers[1:], row_text[1:]):
                clean_v = val.replace('+', '').replace('%', '').strip()
                opp_total[h] = pd.to_numeric(clean_v, errors='coerce')
            continue

        p_id = None
        if link and '/player/' in link.get('href', ''):
            try:
                p_id = int(link.get('href').split('/player/')[1].split('/')[0])
            except (ValueError, IndexError):
                p_id = None

        rec = {'player_id': p_id, 'player_name': row_label}
        for i, (h, val) in enumerate(zip(headers[1:], row_text[1:]), start=1):
            clean_v = val.replace('+', '').replace('%', '').strip()
            rec[h] = pd.to_numeric(clean_v, errors='coerce')

        if min_idx is not None:
            min_cell = cells[min_idx]
            min_tip = parse_min_tooltip(min_cell)
            for k, v in min_tip.items():
                rec[k] = v

        if ofg_idx is not None:
            ofg_cell = cells[ofg_idx]
            tip = parse_ofg_tooltip(ofg_cell)

            ofg_data = tip.get('OFG', {})
            rec['OFG_pct'] = ofg_data.get('pct', rec.get('OFG', np.nan)) if isinstance(ofg_data, dict) else float(ofg_data)
            rec['OFG_M_season'] = ofg_data.get('M', np.nan) if isinstance(ofg_data, dict) else np.nan
            rec['OFG_A_season'] = ofg_data.get('A', np.nan) if isinstance(ofg_data, dict) else np.nan

            o2p = tip.get('O2P', {})
            rec['O2P_pct'] = o2p.get('pct', np.nan) if isinstance(o2p, dict) else float(o2p)
            rec['O2P_M_season'] = o2p.get('M', np.nan) if isinstance(o2p, dict) else np.nan
            rec['O2P_A_season'] = o2p.get('A', np.nan) if isinstance(o2p, dict) else np.nan

            o3p = tip.get('O3P', {})
            rec['O3P_pct'] = o3p.get('pct', np.nan) if isinstance(o3p, dict) else float(o3p)
            rec['O3P_M_season'] = o3p.get('M', np.nan) if isinstance(o3p, dict) else np.nan
            rec['O3P_A_season'] = o3p.get('A', np.nan) if isinstance(o3p, dict) else np.nan

            oefg = tip.get('OeFGP', np.nan)
            rec['OeFGP_pct'] = float(oefg) if not isinstance(oefg, dict) else oefg.get('pct', np.nan)
            odist = tip.get('ODIST', np.nan)
            rec['ODIST_opp'] = float(odist) if not isinstance(odist, dict) else np.nan

        player_records.append(rec)

    return pd.DataFrame(player_records), team_total, opp_total


def engineer_features(df_players: pd.DataFrame, team_total: dict, opp_total: dict) -> pd.DataFrame:
    df = df_players.copy()

    team_fga = team_total.get('FGA', df['FGA'].sum())
    team_fgm = team_total.get('FGM', df['FGM'].sum())
    team_fta = team_total.get('FTA', df['FTA'].sum())
    team_off = team_total.get('Off', df['Off'].sum())
    team_def = team_total.get('Def', df['Def'].sum())
    team_reb = team_total.get('Reb', df['Reb'].sum())
    team_to = team_total.get('TO', df['TO'].sum())
    team_plus = team_total.get('+/-', df['+/-'].sum())

    opp_fga = opp_total.get('FGA', team_fga)
    opp_3pa = opp_total.get('3PA', team_total.get('3PA', 20.0))
    opp_off = opp_total.get('Off', team_off)
    opp_def = opp_total.get('Def', team_def)
    opp_reb = opp_total.get('Reb', team_reb)

    team_poss_est = team_fga + 0.475 * team_fta - team_off + team_to
    if team_poss_est <= 0:
        team_poss_est = 70.0

    player_min = df['Min'].fillna(1.0)
    min_frac = (player_min / 40.0).replace(0, 0.05)
    player_poss_est = (team_poss_est * min_frac).replace(0, 1.0)
    df['Poss_est'] = player_poss_est

    df['AST_P'] = (100.0 * df['Ast'] / (min_frac * team_fgm - df['FGM']).replace(0, 1.0)).clip(0, 80.0)
    df['TO_P'] = (100.0 * df['TO'] / (df['FGA'] + 0.475 * df['FTA'] + df['TO']).replace(0, 1.0)).clip(0, 100.0)
    df['ORB_P'] = (100.0 * df['Off'] / (min_frac * (team_off + opp_def)).replace(0, 1.0)).clip(0, 50.0)
    df['DRB_P'] = (100.0 * df['Def'] / (min_frac * (team_def + opp_off)).replace(0, 1.0)).clip(0, 50.0)
    df['TRB_P'] = (100.0 * df['Reb'] / (min_frac * (team_reb + opp_reb)).replace(0, 1.0)).clip(0, 50.0)
    df['STL_P'] = (100.0 * df['Stl'] / (min_frac * team_poss_est).replace(0, 1.0)).clip(0, 20.0)

    opp_2pa = max(1.0, opp_fga - opp_3pa)
    df['BLK_P'] = (100.0 * df['Blk'] / (min_frac * opp_2pa).replace(0, 1.0)).clip(0, 30.0)
    df['USG_P'] = df['USG%']

    df['PTS_per_56'] = (df['Pts'] / player_poss_est) * 56.0
    df['PF_per_56'] = (df['PF'] / player_poss_est) * 56.0
    df['FGA_per_56'] = (df['FGA'] / player_poss_est) * 56.0
    df['3PA_per_56'] = (df['3PA'] / player_poss_est) * 56.0
    df['FTA_per_56'] = (df['FTA'] / player_poss_est) * 56.0
    df['FD_per_56'] = (df['FD'] / player_poss_est) * 56.0

    true_shots = (df['FGA'] + 0.475 * df['FTA']).replace(0, 1.0)
    df['TS_pct'] = (df['Pts'] / (2.0 * true_shots)) * 100.0

    two_pa = (df['FGA'] - df['3PA']).replace(0, 1.0)
    two_pm = df['FGM'] - df['3PM']
    df['2P_pct'] = (two_pm / two_pa) * 100.0

    df['3P_pct'] = df['3P%']
    df['3PAr'] = df['3PA'] / df['FGA'].replace(0, 1.0)
    df['FTr'] = df['FTA'] / df['FGA'].replace(0, 1.0)
    df['FT_pct'] = df['FT%']
    df['eFG_pct'] = ((df['FGM'] + 0.5 * df['3PM']) / df['FGA'].replace(0, 1.0)) * 100.0

    games = df['G'].replace(0, 1)
    df['O2P_M_per_56'] = (df.get('O2P_M_season', 0) / games / player_poss_est) * 56.0
    df['O2P_A_per_56'] = (df.get('O2P_A_season', 0) / games / player_poss_est) * 56.0
    df['O3P_M_per_56'] = (df.get('O3P_M_season', 0) / games / player_poss_est) * 56.0
    df['O3P_A_per_56'] = (df.get('O3P_A_season', 0) / games / player_poss_est) * 56.0
    total_OA = (df.get('O2P_A_season', 0) + df.get('O3P_A_season', 0)).replace(0, np.nan)
    df['O3PAr_opp'] = df.get('O3P_A_season', 0) / total_OA

    pos_estimates = []
    has_pos_mins = []
    for _, row in df.iterrows():
        pg_m = row.get('PG_Min', 0.0) or 0.0
        sg_m = row.get('SG_Min', 0.0) or 0.0
        sf_m = row.get('SF_Min', 0.0) or 0.0
        pf_m = row.get('PF_Min', 0.0) or 0.0
        c_m = row.get('C_Min', 0.0) or 0.0
        tot_pos_m = pg_m + sg_m + sf_m + pf_m + c_m
        if tot_pos_m > 0:
            pos_dict = {'PG': pg_m, 'SG': sg_m, 'SF': sf_m, 'PF': pf_m, 'C': c_m}
            primary = max(pos_dict, key=pos_dict.get)
            pos_estimates.append(primary)
            has_pos_mins.append(True)
        else:
            ast_val = row.get('Ast', 0)
            reb_val = row.get('Reb', 0)
            blk_val = row.get('Blk', 0)
            dist_val = row.get('DIST', 10.0)
            if ast_val >= 4.0 or (ast_val >= 3.0 and dist_val >= 12.0):
                primary = 'PG'
            elif dist_val >= 12.0 or row.get('3PA', 0) >= 4.0:
                primary = 'SG'
            elif reb_val >= 7.0 or blk_val >= 0.8:
                primary = 'C' if blk_val >= 1.0 or reb_val >= 8.5 else 'PF'
            else:
                primary = 'SF'
            pos_estimates.append(primary)
            has_pos_mins.append(False)

    df['primary_pos_est'] = pos_estimates

    pg_fracs, sg_fracs, sf_fracs, pf_fracs, c_fracs = [], [], [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        if has_pos_mins[i]:
            m_tot = max(0.1, row.get('Min', 1.0))
            pg_fracs.append(min(1.0, (row.get('PG_Min', 0.0) or 0.0) / m_tot))
            sg_fracs.append(min(1.0, (row.get('SG_Min', 0.0) or 0.0) / m_tot))
            sf_fracs.append(min(1.0, (row.get('SF_Min', 0.0) or 0.0) / m_tot))
            pf_fracs.append(min(1.0, (row.get('PF_Min', 0.0) or 0.0) / m_tot))
            c_fracs.append(min(1.0, (row.get('C_Min', 0.0) or 0.0) / m_tot))
        else:
            p = pos_estimates[i]
            pg_fracs.append(1.0 if p == 'PG' else 0.0)
            sg_fracs.append(1.0 if p == 'SG' else 0.0)
            sf_fracs.append(1.0 if p == 'SF' else 0.0)
            pf_fracs.append(1.0 if p == 'PF' else 0.0)
            c_fracs.append(1.0 if p == 'C' else 0.0)

    df['PG_frac'] = pg_fracs
    df['SG_frac'] = sg_fracs
    df['SF_frac'] = sf_fracs
    df['PF_frac'] = pf_fracs
    df['C_frac'] = c_fracs

    df['player_pm_per_poss'] = df['+/-'] / player_poss_est
    df['team_pm_per_poss'] = team_plus / team_poss_est
    df['diff_vs_team_avg'] = df['player_pm_per_poss'] - df['team_pm_per_poss']

    df['box_creation'] = df['AST_P'] * 0.18 + df['PTS_per_56'] * 0.22 + (df['3PM'] / player_poss_est) * 56.0 * 0.35
    df['scoring_load'] = df['PTS_per_56'] * (df['TS_pct'] / 100.0)
    df['def_activity_index'] = (df['STL_P'] * 1.5 + df['BLK_P'] * 1.2) * (np.maximum(0, df['DRB_P']) ** 0.5)
    df['rim_anchor_score'] = df['BLK_P'] * df['DRB_P'] * df['C_frac']
    df['perimeter_creator_score'] = df['AST_P'] * (df['USG%'] / 100.0) * (df['PG_frac'] + df['SG_frac'])
    df['foul_drawing_impact'] = df['FD_per_56'] * df['FTr']

    df['AST_TO_ratio'] = df['AST_P'] / df['TO_P'].replace(0, 0.5)
    df['USG_x_TS'] = df['USG%'] * (df['TS_pct'] / 100.0)
    df['USG_x_AST'] = df['USG%'] * df['AST_P']
    df['BLK_x_DRB'] = df['BLK_P'] * df['DRB_P']
    df['STL_x_AST'] = df['STL_P'] * df['AST_P']

    mapping = {
        'Min': 'Min', 'GP': 'G', 'GS': 'GS', 'PTS': 'Pts', 'FG_M': 'FGM', 'FG_A': 'FGA',
        '_3P_M': '3PM', '_3P_A': '3PA', 'FT_M': 'FTM', 'FT_A': 'FTA', 'Off': 'Off', 'Def': 'Def',
        'Rebs': 'Reb', 'AST': 'Ast', 'STL': 'Stl', 'BLK': 'Blk', 'TO': 'TO', 'PF': 'PF',
        'DIST': 'DIST', 'PITP': 'PITP', 'FBP': 'FBP', 'FD': 'FD'
    }
    for feat_name, orig_col in mapping.items():
        if orig_col in df.columns:
            df[feat_name] = df[orig_col]

    return df


def apply_position_adjustment(df: pd.DataFrame, pos_means: dict, adjust_cols: list) -> pd.DataFrame:
    df_out = df.copy()
    cols_present = [c for c in adjust_cols if c in df_out.columns]
    if not cols_present or not pos_means:
        return df_out

    expected = np.zeros((len(df_out), len(cols_present)))
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        pos_frac = df_out[f'{pos}_frac'].fillna(0.0).values[:, None]
        pos_vec = np.array([pos_means[pos].get(c, 0.0) for c in cols_present])[None, :]
        expected += pos_frac * pos_vec

    df_out[cols_present] = df_out[cols_present].fillna(0.0) - expected
    return df_out


def predict_brapm_dict(soup_or_html) -> dict:
    """
    Given a BeautifulSoup object or HTML string of a team statistics page,
    predicts bORAPM, bDRAPM, and bRAPM for every player on the page.
    Returns:
        dict: {player_id: {"bORAPM": float, "bDRAPM": float, "bRAPM": float, "Pos": str}}
    """
    df_players, team_total, opp_total = parse_team_html_to_df(soup_or_html)
    if df_players.empty:
        return {}

    bundle = load_brapm_bundle()
    df_feat_raw = engineer_features(df_players, team_total, opp_total)

    pos_means = bundle.get('pos_means', None)
    pos_adjust_cols = bundle.get('pos_adjust_cols', [])
    if pos_means and pos_adjust_cols:
        df_feat = apply_position_adjustment(df_feat_raw, pos_means, pos_adjust_cols)
    else:
        df_feat = df_feat_raw

    rate_cols = bundle.get('rate_feature_cols')
    pm_cols = bundle['pm_feature_cols']
    X_rate = df_feat[rate_cols].fillna(0)
    X_pm = df_feat_raw[pm_cols].fillna(0)

    stage1 = bundle.get('stage1_rate_models')
    stage2 = bundle.get('stage2_rate_blend_models')

    raw_rate_o = stage1['oRAPM'].predict(X_rate)
    raw_rate_d = stage1['dRAPM'].predict(X_rate)

    adj_o = stage2['oRAPM'].predict(X_pm) if stage2 else 0.0
    adj_d = stage2['dRAPM'].predict(X_pm) if stage2 else 0.0

    total_poss = df_feat_raw['Poss_est'] * df_players['G'].fillna(30.0)
    rate_shrink_k = 150.0
    rate_shrink_weight = (total_poss / (total_poss + rate_shrink_k)).clip(0.10, 1.0)

    pos_prior_rate_o = {'PG': -0.6, 'SG': -0.6, 'SF': -0.7, 'PF': -0.7, 'C': -0.8}
    pos_prior_rate_d = {'PG': -0.6, 'SG': -0.5, 'SF': -0.4, 'PF': -0.3, 'C': -0.2}
    prior_rate_o = df_feat_raw['primary_pos_est'].map(pos_prior_rate_o).fillna(-0.6)
    prior_rate_d = df_feat_raw['primary_pos_est'].map(pos_prior_rate_d).fillna(-0.4)

    bORAPM = np.round(rate_shrink_weight * (raw_rate_o + adj_o) + (1.0 - rate_shrink_weight) * prior_rate_o, 1)
    bDRAPM = np.round(rate_shrink_weight * (raw_rate_d + adj_d) + (1.0 - rate_shrink_weight) * prior_rate_d, 1)
    bRAPM = np.round(bORAPM + bDRAPM, 1)

    result = {}
    for i, row in df_players.iterrows():
        p_id = row['player_id']
        p_name = row['player_name']
        data = {
            "bORAPM": float(bORAPM[i]),
            "bDRAPM": float(bDRAPM[i]),
            "bRAPM": float(bRAPM[i]),
            "Pos": df_feat_raw['primary_pos_est'].iloc[i]
        }
        if p_id is not None:
            result[p_id] = data
        if p_name:
            result[p_name] = data

    return result
