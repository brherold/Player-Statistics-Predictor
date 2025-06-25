from RecruitSkillPredictorCol.scripts.functions_3 import *
#from functions_3 import *
import pandas as pd
import joblib
import os
from itertools import zip_longest


from bs4 import BeautifulSoup
import requests

hardwood_link = "http://onlinecollegebasketball.org/player/"



def save_html_to_file(content_bytes, filename):
    # Write raw bytes—use wb to overwrite or ab to append
    with open(filename, 'wb') as f:
        f.write(content_bytes)

def load_html_from_file(filename):
    with open(filename, 'rb') as file:
        return file.read()







def CurSkill(soup,is_INT):

    
    columns = {'FR_SI': 0, 'SO_SI': 0, 'JR_SI': 0, 'SR_SI': 0, 'FR-SR-Dev': 0, 'Eval_txt' : "","IS_Eval": 0, "OS_Eval": 0, "RNG_Eval": 0, "Reb_Eval": 0,
            "Def_Eval": 0, "ID_Eval": 0, "PD_Eval": 0, "IQ_Eval": 0,
            "PA_Eval": 0, "HND_Eval": 0, "Spd_Eval": 0, "All_Around_Eval": 0, 'FR_IS': 0, 'FR_OS': 0, 'FR_Rng': 0, 'FR_Fin': 0, 'FR_Reb': 0, 
    'FR_IDef': 0, 'FR_PDef': 0, 'FR_IQ': 0, 'FR_Pass': 0, 'FR_Hnd': 0, 
    'FR_Drv': 0, 'FR_Spd': 0, 'FR_Sta': 0}
    
    

    # Create a single-row DataFrame
    df = pd.DataFrame([columns])

    


    _ , eval_scores = get_eval(soup)
    skill_values = get_skills(soup)
    potential = extract_potential(soup)
    current_strength = extract_current_strength(soup)
    player_class = extract_player_class(soup)
    


   
    

    for key, value in eval_scores.items():
        df.loc[0,key] = value

    df.at[0, 'POT'] = potential

    #For International
    if is_INT == 1:
            #Senior


        df.at[0, 'HS-4_IS'] = skill_values['IS'][0]
        df.at[0, 'HS-4_OS'] = skill_values['OS'][0]
        df.at[0, 'HS-4_Rng'] = skill_values['RNG'][0]
        df.at[0, 'HS-4_Fin'] = skill_values['FIN'][0]
        df.at[0, 'HS-4_Reb'] = skill_values['REB'][0]
        df.at[0, 'HS-4_IDef'] = skill_values['ID'][0]
        df.at[0, 'HS-4_PDef'] = skill_values['PD'][0]
        df.at[0, 'HS-4_IQ'] = skill_values['IQ'][0]
        df.at[0, 'HS-4_Pass'] = skill_values['PASS'][0]
        df.at[0, 'HS-4_Hnd'] = skill_values['HND'][0]
        df.at[0, 'HS-4_Drv'] = skill_values['DRV'][0]
        df.at[0, 'HS-4_Spd'] = skill_values['SPD'][0]
        df.at[0, 'HS-4_Sta'] = skill_values['STA'][0]


        #College True Freshman
        if len(skill_values['SI']) > 2:
            df.at[0, 'Col-1_IS'] = skill_values['IS'][2]
            df.at[0, 'Col-1_OS'] = skill_values['OS'][2]
            df.at[0, 'Col-1_Rng'] = skill_values['RNG'][2]
            df.at[0, 'Col-1_Fin'] = skill_values['FIN'][2]
            df.at[0, 'Col-1_Reb'] = skill_values['REB'][2]
            df.at[0, 'Col-1_IDef'] = skill_values['ID'][2]
            df.at[0, 'Col-1_PDef'] = skill_values['PD'][2]
            df.at[0, 'Col-1_IQ'] = skill_values['IQ'][2]
            df.at[0, 'Col-1_Pass'] = skill_values['PASS'][2]
            df.at[0, 'Col-1_Hnd'] = skill_values['HND'][2]
            df.at[0, 'Col-1_Drv'] = skill_values['DRV'][2]
            df.at[0, 'Col-1_Spd'] = skill_values['SPD'][2]
            df.at[0, 'Col-1_Sta'] = skill_values['STA'][2]

        
            #College 2nd Year
        if len(skill_values['SI']) > 4:

            df.at[0, 'Col-2_IS'] = skill_values['IS'][4]
            df.at[0, 'Col-2_OS'] = skill_values['OS'][4]
            df.at[0, 'Col-2_Rng'] = skill_values['RNG'][4]
            df.at[0, 'Col-2_Fin'] = skill_values['FIN'][4]
            df.at[0, 'Col-2_Reb'] = skill_values['REB'][4]
            df.at[0, 'Col-2_IDef'] = skill_values['ID'][4]
            df.at[0, 'Col-2_PDef'] = skill_values['PD'][4]
            df.at[0, 'Col-2_IQ'] = skill_values['IQ'][4]
            df.at[0, 'Col-2_Pass'] = skill_values['PASS'][4]
            df.at[0, 'Col-2_Hnd'] = skill_values['HND'][4]
            df.at[0, 'Col-2_Drv'] = skill_values['DRV'][4]
            df.at[0, 'Col-2_Spd'] = skill_values['SPD'][4]
            df.at[0, 'Col-2_Sta'] = skill_values['STA'][4]

        # College 3rd Year
        if len(skill_values['SI']) > 6:

            df.at[0, 'Col-3_IS'] = skill_values['IS'][6]
            df.at[0, 'Col-3_OS'] = skill_values['OS'][6]
            df.at[0, 'Col-3_Rng'] = skill_values['RNG'][6]
            df.at[0, 'Col-3_Fin'] = skill_values['FIN'][6]
            df.at[0, 'Col-3_Reb'] = skill_values['REB'][6]
            df.at[0, 'Col-3_IDef'] = skill_values['ID'][6]
            df.at[0, 'Col-3_PDef'] = skill_values['PD'][6]
            df.at[0, 'Col-3_IQ'] = skill_values['IQ'][6]
            df.at[0, 'Col-3_Pass'] = skill_values['PASS'][6]
            df.at[0, 'Col-3_Hnd'] = skill_values['HND'][6]
            df.at[0, 'Col-3_Drv'] = skill_values['DRV'][6]
            df.at[0, 'Col-3_Spd'] = skill_values['SPD'][6]
            df.at[0, 'Col-3_Sta'] = skill_values['STA'][6]


    else:


        df.at[0, 'HS-1_IS'] = skill_values['IS'][0]
        df.at[0, 'HS-1_OS'] = skill_values['OS'][0]
        df.at[0, 'HS-1_Rng'] = skill_values['RNG'][0]
        df.at[0, 'HS-1_Fin'] = skill_values['FIN'][0]
        df.at[0, 'HS-1_Reb'] = skill_values['REB'][0]
        df.at[0, 'HS-1_IDef'] = skill_values['ID'][0]
        df.at[0, 'HS-1_PDef'] = skill_values['PD'][0]
        df.at[0, 'HS-1_IQ'] = skill_values['IQ'][0]
        df.at[0, 'HS-1_Pass'] = skill_values['PASS'][0]
        df.at[0, 'HS-1_Hnd'] = skill_values['HND'][0]
        df.at[0, 'HS-1_Drv'] = skill_values['DRV'][0]
        df.at[0, 'HS-1_Spd'] = skill_values['SPD'][0]
        df.at[0, 'HS-1_Sta'] = skill_values['STA'][0]

        
        #Sophmore
        if len(skill_values['SI']) > 2:
        

            df.at[0, 'HS-2_IS'] = skill_values['IS'][2]
            df.at[0, 'HS-2_OS'] = skill_values['OS'][2]
            df.at[0, 'HS-2_Rng'] = skill_values['RNG'][2]
            df.at[0, 'HS-2_Fin'] = skill_values['FIN'][2]
            df.at[0, 'HS-2_Reb'] = skill_values['REB'][2]
            df.at[0, 'HS-2_IDef'] = skill_values['ID'][2]
            df.at[0, 'HS-2_PDef'] = skill_values['PD'][2]
            df.at[0, 'HS-2_IQ'] = skill_values['IQ'][2]
            df.at[0, 'HS-2_Pass'] = skill_values['PASS'][2]
            df.at[0, 'HS-2_Hnd'] = skill_values['HND'][2]
            df.at[0, 'HS-2_Drv'] = skill_values['DRV'][2]
            df.at[0, 'HS-2_Spd'] = skill_values['SPD'][2]
            df.at[0, 'HS-2_Sta'] = skill_values['STA'][2]

        
        
        #Junior
        if len(skill_values['SI']) > 4:

            df.at[0, 'HS-3_IS'] = skill_values['IS'][4]
            df.at[0, 'HS-3_OS'] = skill_values['OS'][4]
            df.at[0, 'HS-3_Rng'] = skill_values['RNG'][4]
            df.at[0, 'HS-3_Fin'] = skill_values['FIN'][4]
            df.at[0, 'HS-3_Reb'] = skill_values['REB'][4]
            df.at[0, 'HS-3_IDef'] = skill_values['ID'][4]
            df.at[0, 'HS-3_PDef'] = skill_values['PD'][4]
            df.at[0, 'HS-3_IQ'] = skill_values['IQ'][4]
            df.at[0, 'HS-3_Pass'] = skill_values['PASS'][4]
            df.at[0, 'HS-3_Hnd'] = skill_values['HND'][4]
            df.at[0, 'HS-3_Drv'] = skill_values['DRV'][4]
            df.at[0, 'HS-3_Spd'] = skill_values['SPD'][4]
            df.at[0, 'HS-3_Sta'] = skill_values['STA'][4]


        
        #Senior
        if len(skill_values['SI']) > 6:


            df.at[0, 'HS-4_IS'] = skill_values['IS'][6]
            df.at[0, 'HS-4_OS'] = skill_values['OS'][6]
            df.at[0, 'HS-4_Rng'] = skill_values['RNG'][6]
            df.at[0, 'HS-4_Fin'] = skill_values['FIN'][6]
            df.at[0, 'HS-4_Reb'] = skill_values['REB'][6]
            df.at[0, 'HS-4_IDef'] = skill_values['ID'][6]
            df.at[0, 'HS-4_PDef'] = skill_values['PD'][6]
            df.at[0, 'HS-4_IQ'] = skill_values['IQ'][6]
            df.at[0, 'HS-4_Pass'] = skill_values['PASS'][6]
            df.at[0, 'HS-4_Hnd'] = skill_values['HND'][6]
            df.at[0, 'HS-4_Drv'] = skill_values['DRV'][6]
            df.at[0, 'HS-4_Spd'] = skill_values['SPD'][6]
            df.at[0, 'HS-4_Sta'] = skill_values['STA'][6]

        #College True Freshman
        if len(skill_values['SI']) > 8:


            df.at[0, 'Col-1_IS'] = skill_values['IS'][8]
            df.at[0, 'Col-1_OS'] = skill_values['OS'][8]
            df.at[0, 'Col-1_Rng'] = skill_values['RNG'][8]
            df.at[0, 'Col-1_Fin'] = skill_values['FIN'][8]
            df.at[0, 'Col-1_Reb'] = skill_values['REB'][8]
            df.at[0, 'Col-1_IDef'] = skill_values['ID'][8]
            df.at[0, 'Col-1_PDef'] = skill_values['PD'][8]
            df.at[0, 'Col-1_IQ'] = skill_values['IQ'][8]
            df.at[0, 'Col-1_Pass'] = skill_values['PASS'][8]
            df.at[0, 'Col-1_Hnd'] = skill_values['HND'][8]
            df.at[0, 'Col-1_Drv'] = skill_values['DRV'][8]
            df.at[0, 'Col-1_Spd'] = skill_values['SPD'][8]
            df.at[0, 'Col-1_Sta'] = skill_values['STA'][8]
        
            #College 2nd Year
        if len(skill_values['SI']) > 10:


            df.at[0, 'Col-2_IS'] = skill_values['IS'][10]
            df.at[0, 'Col-2_OS'] = skill_values['OS'][10]
            df.at[0, 'Col-2_Rng'] = skill_values['RNG'][10]
            df.at[0, 'Col-2_Fin'] = skill_values['FIN'][10]
            df.at[0, 'Col-2_Reb'] = skill_values['REB'][10]
            df.at[0, 'Col-2_IDef'] = skill_values['ID'][10]
            df.at[0, 'Col-2_PDef'] = skill_values['PD'][10]
            df.at[0, 'Col-2_IQ'] = skill_values['IQ'][10]
            df.at[0, 'Col-2_Pass'] = skill_values['PASS'][10]
            df.at[0, 'Col-2_Hnd'] = skill_values['HND'][10]
            df.at[0, 'Col-2_Drv'] = skill_values['DRV'][10]
            df.at[0, 'Col-2_Spd'] = skill_values['SPD'][10]
            df.at[0, 'Col-2_Sta'] = skill_values['STA'][10]
        
        #College 3rd Year
        if len(skill_values['SI']) > 12:

            df.at[0, 'Col-3_IS'] = skill_values['IS'][12]
            df.at[0, 'Col-3_OS'] = skill_values['OS'][12]
            df.at[0, 'Col-3_Rng'] = skill_values['RNG'][12]
            df.at[0, 'Col-3_Fin'] = skill_values['FIN'][12]
            df.at[0, 'Col-3_Reb'] = skill_values['REB'][12]
            df.at[0, 'Col-3_IDef'] = skill_values['ID'][12]
            df.at[0, 'Col-3_PDef'] = skill_values['PD'][12]
            df.at[0, 'Col-3_IQ'] = skill_values['IQ'][12]
            df.at[0, 'Col-3_Pass'] = skill_values['PASS'][12]
            df.at[0, 'Col-3_Hnd'] = skill_values['HND'][12]
            df.at[0, 'Col-3_Drv'] = skill_values['DRV'][12]
            df.at[0, 'Col-3_Spd'] = skill_values['SPD'][12]
            df.at[0, 'Col-3_Sta'] = skill_values['STA'][12]



    return df,current_strength, player_class

#Load Models

#year (1,2,3,4) -> HS YEAR that its look at for dev/skill
#predYear (1,2,3,4,5) -> predicted COL YEAR skill
def getPredictedSkills(playerID,predYear):
    #Check if player Predict Exists
    # Prepare save folder and filename

    folder = "RecruitSkillPredictorCol/PlayerPredicts"
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f'{playerID}-{predYear}-playerSkils-2046.pkl')

    # If prediction already saved, load and return it
    if os.path.exists(save_path):
        predicted_skills = joblib.load(save_path)
        
        return predicted_skills

    file_name = f"{playerID}.html"
    folder_name = "RecruitSkillPredictorCol/Players-Checked-HTML"  # Corrected folder name

    player_link = hardwood_link + str(playerID)

    try:
        #checks if Player is already in PlayersHTML folder
        html_content = load_html_from_file(f"{folder_name}/{file_name}")
        #print("Content loaded from file.")
    except FileNotFoundError:
        response = requests.get(player_link + "/D")
        html_content = response.content
        os.makedirs(folder_name,
                    exist_ok=True)  # Create folder if it doesn't exist
        save_html_to_file(html_content, f"{folder_name}/{file_name}")
        #print("Content fetched from URL and saved to file.")

    soup = BeautifulSoup(html_content, "html.parser")

    country = get_country(soup)

    #Checks if player is International
    if country not in ["United States", "Canada", "Puerto Rico"]:
        is_INT = 1
    else:
        is_INT = 0



    df, current_strength, player_class = CurSkill(soup,is_INT)

    

    #Most recent dev class
    if player_class > 7:
        year = 7
    else:
        year = player_class
    
    

    model, expected_columns= load_model_components(predYear, year,is_INT)

    X = df[expected_columns]

    
    
    prediction = model.predict(X)[0]

    
    target_names = ['IS', 'OS', 'Rng', 'Fin', 'Reb', 'IDef',
            'PDef', 'IQ', 'Pass', 'Hnd', 'Drv',  'Spd', 'Sta']

        
    


    # Split at "IQ"
    split_index = target_names.index('IQ')

    col1_names = target_names[:split_index]
    col2_names = target_names[split_index:]
    col1_preds = prediction[:split_index]
    col2_preds = prediction[split_index:]

    predicted_skills = {}
    #predicted_skills["Class"] = player_class
    predicted_skills["Str"] = current_strength

    for name1, pred1, name2, pred2 in zip_longest(col1_names, col1_preds, col2_names, col2_preds, fillvalue=""):
        if name1 and pred1 != "":
            predicted_skills[name1] = round(float(pred1))
        if name2 and pred2 != "":
            predicted_skills[name2] = round(float(pred2))

    # Save to predicted skills for particular player in current season
    joblib.dump(predicted_skills, save_path)


    return predicted_skills



#print(getPredictedSkills(237714,5))