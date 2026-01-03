import pandas as pd
from collections import defaultdict


def build_stat_distributions(
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
         'BLK_P', 'TO_P', 'USG_P']

distributions = build_stat_distributions(
    csv_path="DataCSVS/44-45-46-per56.csv",
    stats=stats
)

#print(distributions[("PG", "AST_P")])
'''