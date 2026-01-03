#DONT USE .44 when calculating possessions or true shooting USE .48 (similar to KenPom .475)
from bs4 import BeautifulSoup
import requests
from .predictEPM import *
from tabulate import tabulate
from .StatDistributionGetter import *
import numpy as np

column_stats = ['Primary_Position','PTS', 'O_eFG_P','OEPM', 'DEPM', 'EPM', 'TS', 
         '_3PAr', 'FTr', 'ORB_P', 'DRB_P', 'TRB_P', 'AST_P', 'STL_P', 
         'BLK_P', 'TO_P', 'USG_P']

distributions = build_stat_distributions(
    csv_path="DataCSVS/44-45-46-per56.csv",
    stats=column_stats
)

def get_percentile(value, sorted_values):
    if len(sorted_values) == 0:
        return np.nan
    pct = np.searchsorted(sorted_values, value, side="right") / len(sorted_values) * 100
    return int(min(max(pct, 0), 100))

#Percentile Color
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

def lower_is_better(percentile):
    return 100 - percentile

#EPM COefficient to team WP (Winning Percentage) is .0484 (.05) (r2 = 0.8706  and NetRtg  coeff = 3.8604 r2= .9469
#Player Earned Wins (over replacement level player) = (VORP * .0484 * (Team Games Played))/5

#team_EPM (no plus) coefficient to WP is .0202 (r2 = .8899) and NetRtg is 1.5848 (r2= .9742)
#gameType: A (All), C(Conference), P (Playoff), T(Tournament)
def get_team_player_stats(team_stat_html):

    
    soup = BeautifulSoup(team_stat_html, "html.parser")
    
    header_text = soup.find("h1").text #good enough to just show

    
    
    #'''
    #Gets team_ID through scraping this page
    main_hrefs_arr = soup.find('div', id="Main").find_all("a", href=True)
    record_hrefs = [a['href'] for a in main_hrefs_arr if a.get('href', '').startswith('/records/0/0/')]
    team_id = int(record_hrefs[0].split("/")[-1])
    
    season = int(header_text.split(" ")[0])
    #'''

    
    table = soup.find("table")

    team_stats_thead = table.find_all("thead")[1]

    
    team = team_stats_thead.find_all("th")

    
    team_Min = float(team[3].text) / 5

    team_GP = float(team[1].text)

    team_shots_split = team[6].get("title").replace("\n"," ").split(" ")

    team_FG_M , team_FG_A = map(int,team_shots_split[3].strip("()").split("-"))

    team_FG_M = float(team_FG_M / team_GP)
    team_FG_A = float(team_FG_A / team_GP)

    
    team_FT_split = team[12].get("title").split(" ")

    _, team_FT_A = map(int,team_FT_split[-1].strip("()").split("-"))

    team_FT_A = float(team_FT_A / team_GP)

    team_PTS = float(team[13].text)
    team_Off = float(team[17].text)
    team_Def = float(team[18].text)
    team_Rebs = team_Off + team_Def

    team_Ast = float(team[20].text)
    team_Stl = float(team[21].text)
    team_TO = float(team[24].text)
    team_PF = float(team[25].text)
    team_FD = float(team[27].text)

    #Possessions
    team_Poss = team_FG_A - team_Off + team_TO + .48*(team_FT_A) #Will make Opp_Poss same too
    

    opponent_stats_thead = table.find_all("thead")[-1]
    
    opponent = opponent_stats_thead.find_all("th")

    opp_PTS = float(opponent[13].text)

    opp_shots_split = opponent[6].get("title").replace("\n"," ").split(" ")

    opp_FG_M , opp_FG_A = map(int,opp_shots_split[3].strip("()").split("-"))

    opp_FG_M = float(opp_FG_M / team_GP)
    opp_FG_A = float(opp_FG_A / team_GP)
    
    
    opp_3P_M , opp_3P_A = map(int,opp_shots_split[11].strip("()").split("-"))
    

    opp_3P_M = float(opp_3P_M / team_GP)
    opp_3P_A = float(opp_3P_A / team_GP)
    ####
    opp_FT_split = opponent[12].get("title").split(" ")

    _, opp_FT_A = map(int,opp_FT_split[-1].strip("()").split("-"))

    opp_FT_A = float(opp_FT_A / team_GP)

    opp_Off = float(opponent[17].text)
    opp_Def = float(opponent[18].text)
    opp_Rebs = opp_Off + opp_Def

    opp_TO = float(opponent[24].text)

    #Possessions
    opp_Poss = opp_FG_A - opp_Off + opp_TO + .48*(opp_FT_A) #Will make Opp_Poss same too


    #off_values = [Poss,Pts,FGA,FTA,Off,AST,TO,FD]
    #def_values = [OPoss,OPTS,OFGA,OFTA,DReb,STL,PF]



    

    team_ORtg = round(float(team_PTS/team_Poss) * 100 ,1)
    team_DRtg = round(float(opp_PTS/opp_Poss) * 100,1)


    team_stats = (team_GP, round(team_Poss,1), team_ORtg, team_DRtg, round(team_ORtg - team_DRtg,1))
    player_stats = table.find_all("tr")[1:-4]

    player_stats_result = []
    #####
    
    for player in player_stats:
   
    #
    #player = table.find_all("tr")[7] #Trent Shaffer

        player_name = player.find_all("td")[0].text
        

        player_id = player.find_all("td")[0].find("a")["href"].split("/")[-1]

        #Checker
        '''
        if int(player_id) != 180696:
            continue 
        s
        '''

        player_GP = int(player.find_all("td")[1].text)

        if player_GP == 0:
            player_GP = 1
        
        player_GS = int(player.find_all("td")[2].text)
            

        player_Min_td = player.find_all("td")[3]
        
        player_Min = float(player_Min_td.text)
        if player_Min == 0:
            player_Min = 1
        
        player_Min_td_title_split = player_Min_td.get("title").replace("\n"," ").split(" ")

        player_Min_PG = float(player_Min_td_title_split[1])
        player_Min_SG = float(player_Min_td_title_split[4])
        player_Min_SF = float(player_Min_td_title_split[7])
        player_Min_PF = float(player_Min_td_title_split[10])
        player_Min_C = float(player_Min_td_title_split[13])

        player_shots_split = player.find_all("td")[6].get("title").replace("\n"," ").split(" ")

        player_FG_M , player_FG_A = map(int,player_shots_split[3].strip("()").split("-"))
        player_2P_M , _ = map(int,player_shots_split[7].strip("()").split("-"))
        player_3P_M , player_3P_A = map(int,player_shots_split[11].strip("()").split("-"))

        player_FG_M = float(player_FG_M / player_GP)
        player_FG_A = float(player_FG_A / player_GP)
        player_2P_M =  float(player_2P_M / player_GP)
        player_3P_A = float(player_3P_A / player_GP)
        player_3P_M =  float(player_3P_M / player_GP)
        
        
        
        player_FT_split = player.find_all("td")[12].get("title").split(" ")

        player_FT_M, player_FT_A = map(int,player_FT_split[-1].strip("()").split("-"))

        player_FT_M = float(player_FT_M / player_GP)

        player_FT_A = float(player_FT_A/ player_GP)

        player_PTS = float(player.find_all("td")[13].text)
        #ORB
        player_Off = float(player.find_all("td")[17].text)
        player_Def = float(player.find_all("td")[18].text)
        player_Rebs = player_Off + player_Def

        player_Ast = float(player.find_all("td")[20].text)
        player_Stl = float(player.find_all("td")[21].text)
        player_Blk = float(player.find_all("td")[22].text)
        player_TO = float(player.find_all("td")[24].text)
        player_PF = float(player.find_all("td")[25].text)
        player_FD = float(player.find_all("td")[27].text)
        
        ###    

        player_opp_shots_split = player.find_all("td")[23].get("title").replace("\n"," ").split(" ")

        player_O_FG_M , player_O_FG_A = map(int,player_opp_shots_split[3].strip("()").split("-"))
        player_O_2P_M , _ = map(int,player_opp_shots_split[7].strip("()").split("-"))
        player_O_3P_M , _ = map(int,player_opp_shots_split[11].strip("()").split("-"))

       
        player_O_FG_M = float(player_O_FG_M / player_GP)
        player_O_FG_A = float(player_O_FG_A / player_GP)
        player_O_2P_M =  float(player_O_2P_M / player_GP)
        player_O_3P_M =  float(player_O_3P_M / player_GP)
        
        player_O_eFG = float((player_O_FG_M + .5 * player_O_3P_M) / player_O_FG_A) if player_O_FG_A != 0 else "-"

        player_O_PTS = float(player_O_2P_M * 2 + player_O_3P_M * 3)

        

        player_Poss = float( team_Poss * player_Min/ team_Min)
        
        
        
        

        player_off_values_epm = [player_FG_A, player_2P_M, player_3P_M, player_FT_M, player_Ast, player_TO, player_Off]
        player_def_values_epm = [player_O_FG_A, player_O_2P_M, player_O_3P_M, player_Stl, player_PF, player_Def]
        
        player_off_values_bpm= [player_PTS, player_FG_A, player_FT_A, player_Off, player_Ast, player_TO, player_FD]
        player_def_values_bpm = [player_O_PTS, player_O_FG_A, player_Def, player_Stl, player_PF]
        #print(predict_epm("SG",player_Poss, player_Poss, player_off_values_epm,player_def_values_epm))
        

        PG_epm = predict_epm("PG",player_Poss, player_Poss, player_off_values_epm, player_def_values_epm)
        SG_epm = predict_epm("SG",player_Poss, player_Poss, player_off_values_epm,player_def_values_epm)
        SF_epm = predict_epm("SF",player_Poss, player_Poss, player_off_values_epm,player_def_values_epm)
        PF_epm = predict_epm("PF",player_Poss, player_Poss, player_off_values_epm,player_def_values_epm)
        C_epm = predict_epm("C",player_Poss, player_Poss, player_off_values_epm,player_def_values_epm)

        

        pg_weight, sg_weight, sf_weight, pf_weight, c_weight = player_Min_PG/player_Min, player_Min_SG/player_Min, player_Min_SF/player_Min, player_Min_PF/player_Min, player_Min_C/player_Min  

    
        result_epm = tuple(
            round(a*pg_weight + b*sg_weight + c*sf_weight + d*pf_weight + e*c_weight, 1)
            for a, b, c, d, e in zip(PG_epm, SG_epm, SF_epm, PF_epm, C_epm)
        )


        weights = {
            "PG": pg_weight,
            "SG": sg_weight,
            "SF": sf_weight,
            "PF": pf_weight,
            "C": c_weight
        }

        max_Position = max(weights, key=weights.get)
 
        #print(player_name, max_Position, player_GP, player_Min, result)

        VORP_EPM = round((result_epm[-1] + 3) * (player_Min / (team_Min)) * (player_GP / team_GP),1)

        #Advanced Statistics
        player_TS = round((player_PTS) / (2 *(player_FG_A + .44 * player_FT_A)), 3) if (player_FG_A + .48 * player_FT_A) != 0 else 0 
        player_3PAr = round(player_3P_A / player_FG_A, 3) if player_FG_A != 0 else 0 
        player_FTr = round(player_FT_A / player_FG_A, 3) if player_FG_A != 0 else 0

        denom = player_Min * (team_Off + opp_Def)
        
        player_ORB_P = round(100 * (player_Off * (team_Min)) / denom, 1) if denom != 0 else 0

        denom = player_Min * (team_Def + opp_Off)
        player_DRB_P = round(100 * (player_Def * (team_Min)) / denom, 1) if denom != 0 else 0

        denom = player_Min * (team_Rebs+ opp_Rebs)
        player_TRB_P = round(100 * (player_Rebs * (team_Min)) / denom, 1) if denom != 0 else 0

        denom = ((player_Min / (team_Min)) * team_FG_M) - player_FG_M
        player_AST_P = round(100 * player_Ast / denom, 1) if denom != 0 else 0

        denom = player_Min * opp_Poss
        player_STL_P = round(100 * (player_Stl * (team_Min)) / denom, 1) if denom != 0 else 0

        denom = player_Min * (opp_FG_A - opp_3P_A)
        player_BLK_P = round(100 * (player_Blk * (team_Min)) / denom, 1) if denom != 0 else 0

        denom = player_FG_A + 0.44 * player_FT_A + player_TO
        player_TO_P = round(100 * player_TO / denom, 1) if denom != 0 else 0

        denom = player_Min * (team_FG_A + 0.44 * team_FT_A + team_TO)
        player_USG_P = round(100 * ((player_FG_A + 0.44 * player_FT_A + player_TO) * (team_Min)) / denom, 1) if denom != 0 else 0


        player_PTS_per56 = round(float(player_PTS / player_Poss) * 56,1)


        #VORP_for_EW = round((result[-1] + 3) * (player_Min / (team_Min * 5)) * (player_GP / team_GP),3)
        #EW = round(VORP_for_EW * .0484 * (team_GP),3)
        
        player_OPM = result_epm[0]
        player_DPM = result_epm[1]
        player_EPM = result_epm[2]

        player_dic = {}   
        player_dic["Name"] = player_name
        player_dic["playerID"] = player_id
        player_dic["Position"] = max_Position
        player_dic["GS"] = player_GS
        player_dic["GP"] = player_GP
        player_dic["Min"] = player_Min
        player_dic["OPM"] = {"value": player_OPM ,"percentile": get_percentile(player_OPM, distributions[(max_Position, "OEPM")]),"color": percentile_to_rgb(get_percentile(player_OPM, distributions[(max_Position, "OEPM")]))}
        player_dic["DPM"] = {"value": player_DPM ,"percentile": get_percentile(player_DPM, distributions[(max_Position, "DEPM")]),"color": percentile_to_rgb(get_percentile(player_DPM, distributions[(max_Position, "DEPM")]))}
        player_dic["EPM"] = {"value": player_EPM ,"percentile": get_percentile(player_EPM, distributions[(max_Position, "EPM")]),"color": percentile_to_rgb(get_percentile(player_EPM, distributions[(max_Position, "EPM")]))}
        player_dic["VORP"] = VORP_EPM
        player_dic["PTS_per56"] = {"value": player_PTS_per56 ,"percentile": get_percentile(player_PTS_per56, distributions[(max_Position, "PTS")]),"color": percentile_to_rgb(get_percentile(player_PTS_per56, distributions[(max_Position, "PTS")]))}
        player_dic["TS"] = {"value": player_TS ,"percentile": get_percentile(player_TS, distributions[(max_Position, "TS")]),"color": percentile_to_rgb(get_percentile(player_TS, distributions[(max_Position, "TS")]))}
        player_dic["ThreePAr"] = {"value": player_3PAr ,"percentile": get_percentile(player_3PAr, distributions[(max_Position, "_3PAr")]),"color": percentile_to_rgb(get_percentile(player_3PAr, distributions[(max_Position, "_3PAr")]))}
        player_dic["FTr"] = {"value": player_FTr ,"percentile": get_percentile(player_FTr, distributions[(max_Position, "FTr")]),"color": percentile_to_rgb(get_percentile(player_FTr, distributions[(max_Position, "FTr")]))}
        player_dic["ORB_P"] = {"value": player_ORB_P ,"percentile": get_percentile(player_ORB_P, distributions[(max_Position, "ORB_P")]),"color": percentile_to_rgb(get_percentile(player_ORB_P, distributions[(max_Position, "ORB_P")]))}
        player_dic["DRB_P"] = {"value": player_DRB_P ,"percentile": get_percentile(player_DRB_P, distributions[(max_Position, "DRB_P")]),"color": percentile_to_rgb(get_percentile(player_DRB_P, distributions[(max_Position, "DRB_P")]))}
        player_dic["TRB_P"] = {"value": player_TRB_P ,"percentile": get_percentile(player_TRB_P, distributions[(max_Position, "TRB_P")]),"color": percentile_to_rgb(get_percentile(player_TRB_P, distributions[(max_Position, "TRB_P")]))}
        player_dic["AST_P"] = {"value": player_AST_P ,"percentile": get_percentile(player_AST_P, distributions[(max_Position, "AST_P")]),"color": percentile_to_rgb(get_percentile(player_AST_P, distributions[(max_Position, "AST_P")]))}
        player_dic["TO_P"] = {"value": player_TO_P ,"percentile": lower_is_better(get_percentile(player_TO_P, distributions[(max_Position, "TO_P")])),"color": percentile_to_rgb(lower_is_better(get_percentile(player_TO_P, distributions[(max_Position, "TO_P")])))}
        player_dic["STL_P"] = {"value": player_STL_P ,"percentile": get_percentile(player_STL_P, distributions[(max_Position, "STL_P")]),"color": percentile_to_rgb(get_percentile(player_STL_P, distributions[(max_Position, "STL_P")]))}
        player_dic["BLK_P"] = {"value": player_BLK_P ,"percentile": get_percentile(player_BLK_P, distributions[(max_Position, "BLK_P")]),"color": percentile_to_rgb(get_percentile(player_BLK_P, distributions[(max_Position, "BLK_P")]))}
        player_dic["O_eFG_P"] = {"value": player_O_eFG ,"percentile": lower_is_better(get_percentile(player_O_eFG, distributions[(max_Position, "O_eFG_P")])),"color": percentile_to_rgb(lower_is_better(get_percentile(player_O_eFG, distributions[(max_Position, "O_eFG_P")])))}
        player_dic["USG_P"] = {"value": player_USG_P ,"percentile": get_percentile(player_USG_P, distributions[(max_Position, "USG_P")]),"color": percentile_to_rgb(get_percentile(player_USG_P, distributions[(max_Position, "USG_P")]))}



        player_stats_result.append(player_dic)
        




    #print(player_stats_result)
    
    
    #print(header_text, team_id, season,  team_stats, player_stats_result)
    
    return header_text, team_id, season,  team_stats, player_stats_result
    #print(player_stats_result)

    return 

#get_team_player_stats(409 ,2049,"C") 
