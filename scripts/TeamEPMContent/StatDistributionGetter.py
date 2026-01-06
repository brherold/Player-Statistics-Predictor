import pandas as pd
from collections import defaultdict


def build_stat_player_distributions(
    csv_path,
    stats,
    positions=("PG", "SG", "SF", "PF", "C"),
):
    """
    Builds stat distributions by (position, stat).

    Returns:
        dict[(position, stat)] -> sorted list of values
    """

    df = pd.read_csv(csv_path)

    # Basic eligibility filtering
    df = df[
        (df["Primary_Position"].isin(positions))
    ]
    
    distributions = defaultdict(list)

    for pos in positions:
        pos_df = df[df["Primary_Position"] == pos]

        for stat in stats:
            if stat not in pos_df.columns:
                continue


            if stat == "_3P_P":
                df_3P_A = df[df["_3P_A"] >= 3]
                
                _3P_values = df_3P_A[stat].dropna().tolist()

                if _3P_values:
                    _3P_values.sort()
                    distributions[stat] = _3P_values
            
            elif stat == "FT_P":
                df_FT_A = df[df["FT_A"] >= 3]
                
                FT_values = df_FT_A[stat].dropna().tolist()

                if FT_values:
                    FT_values.sort()
                    distributions[stat] = FT_values

            else:

                values = pos_df[stat].dropna().tolist()

                if values:
                    values.sort()
                    distributions[(pos, stat)] = values

    return distributions
'''
df = pd.read_csv("DataCSVS/44-45-46-per56.csv")

#print(df.columns.tolist())

stats = ['Primary_Position','PTS', 'O_eFG_P','OEPM', 'DEPM', 'EPM', 'TS', 
         '_3PAr', 'FTr', 'ORB_P', 'DRB_P', 'TRB_P', 'AST_P', 'STL_P', 
         'BLK_P', 'TO_P', 'USG_P','_2P_P', '_3P_P', 'FT_P']

player_distributions = build_stat_player_distributions(
    csv_path="DataCSVS/44-45-46-per56.csv",
    stats=stats
)

#print(player_distributions[("PG", "AST_P")])
#print(player_distributions['_3P_P'])
#print(player_distributions['FT_P'])
'''


def build_stat_team_distributions(
    csv_path,
    stats
):
    """
    Builds stat distributions by (position, stat).

    Returns:
        dict[(position, stat)] -> sorted list of values
    """

    df = pd.read_csv(csv_path)

    # Basic eligibility filtering

    distributions = defaultdict(list)


    for stat in stats:
        if stat not in df.columns:
            continue

        values = df[stat].dropna().tolist()

        if values:
            values.sort()
            distributions[stat] = values

    return distributions

'''
df = pd.read_csv("DataCSVS/44-45-46-teamAvg.csv")

print(df.columns.tolist())



stats = ['eFG_P', 'FT_P', '_2P_P', '_3P_P','Pace', '_3PAr','FTr', 'TO_P', 'ORB_P', 'DRB_P', 
         'Pace', 'PITP%', 'ORtg', 'NetRtg']

team_distributions = build_stat_team_distributions(
    csv_path="DataCSVS/44-45-46-teamAvg.csv",
    stats=stats
)
print(team_distributions["PITP%"])
'''