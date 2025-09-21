import pandas as pd
import numpy
from bs4 import BeautifulSoup
import requests
import re
import ast
import joblib


def load_model_components(predYear,year,is_INT):
    year = float(year)
    if is_INT != 1:
        return joblib.load(f'RecruitSkillPredictorCol/Models-New-Def/Col-{predYear}-{year}.pkl')
    else:
        return joblib.load(f'RecruitSkillPredictorCol/Models-INT-New-Def/Col-{predYear}-{year}-INT.pkl') 

#Gets Potential of Player
def extract_potential(soup):
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            row_text = row.get_text(strip=True)
            if "Potential" in row_text:
                potential = row_text.split(":")[-1]


                return int(potential)
    return None  # If not found

#Get Current Strength of Player
def extract_current_strength(soup):
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            row_text = row.get_text(strip=True)
            if "Strength" in row_text:
                row_text = row_text.replace("↓", "").replace("↑", "")
                strength = row_text.split(":")[-1]


                return int(strength)
    return None  # If not found



#Get Class of Player
def extract_player_class(soup):
    class_year_map = {
        "High School Freshman": 1,
        "High School Sophomore": 2,
        "High School Junior": 3,
        "High School Senior": 4,
        "International Prospect": 4,

        "Freshman": 5,
        "Redshirt Freshman": 6,
        "Sophomore": 6,  # same as RS Freshman
        "Redshirt Sophomore": 7,
        "Junior": 7,      # same as RS Sophomore
        "Redshirt Junior": 8,
        "Senior": 8,      # same as RS Junior
        "Redshirt Senior": 9,
        "Graduated": 10,
        "Professional": 10,
        "Retired":10
    }
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            row_text = row.get_text(strip=True)
            if "Class" in row_text:
                match = re.search(r"Class:\s*(.*?)\s{2,}", row_text)
                if match:
                    player_class = match.group(1)
                    
                    player_year = class_year_map[player_class]

                    return player_year
    return None  # If not found


def get_player_name(soup):
    pageList = soup.find("th").find_all("a")[1:]

    player_name = pageList[0].find("img")["title"].replace("Bookmark ","")

    return player_name

#Get Country of Player

def get_country(soup):

    #For getting Country tag (differing visual versions of game (colored vs standard))
    country = soup.find("table", class_="player-card").find_all("img")[0].get("title")

    return country

#Get Dev and SI of player
def get_skills(soup):
    skills_dic = {}

    scripts = soup.find_all('script', {'type': 'text/javascript'})
    draw_chart_scripts = [script for script in scripts if 'function drawChart' in script.text]

    for index, draw_chart_script in enumerate(draw_chart_scripts):
        match = re.search(r'arrayToDataTable\((\[.*?\])\);', draw_chart_script.text, re.DOTALL)
        if match:
            array_text = match.group(1)
            # Convert JavaScript array to Python list  
            try:
                data_array = ast.literal_eval(array_text)
                #Adds skills into skills_dic
            
                label = data_array[0]# Holds first element (the labels )
                #Gets skill names and puts into array for indexing
                skill_names = label[1:]
                for index, skill_name in enumerate(skill_names):
                    skills_dic[skill_name] = []

                #Match data values with skill name 
                for data in data_array[1:]:
                    values = data[1:]
                    skills_amount = len(skill_names)
                    
                    #Append skills into their skill_dic slots
                    for i in range(skills_amount):
                        skills_dic[skill_names[i]].append( values[i])
                        


                        
            except Exception as e:
                print("Error parsing array:", e)
        else:    
            print("No arrayToDataTable found.")

    #Skip every 2 to go to next year
    
    return skills_dic


#skills_dic = {"SI": [], "IS": [], "OS": [], "RNG": [], "FIN": [], "REB": [], "ID": [], "PD": [], "IQ": [], "PASS": [], "HAN": [], "DRV": [], "STR": [], "SPD": [], "STA" : [], "Height": [] }
#skill_names = ['SI', 'IS', 'OS', 'RNG', 'FIN', 'ID', 'PD', 'REB', 'PASS', 'HND', 'DRV', 'IQ', 'STR', 'SPD', 'STA', 'PG', 'SG', 'SF', 'PF', 'C', 'Height']

#Find Rev Eval Text

def extract_recruiting_eval(soup):
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            row_text = row.get_text(strip=True)
            if "Recruiting Evaluation:" in row_text:
                return row_text
    return None  # If not found


def get_eval(soup):
    try:
        rec_eval = extract_recruiting_eval(soup)
    except:
        return "No Recruting Eval found"

    InsideShooting = {
        'Elite': {"Could be a dominating post player"},
        
        "Great": {
            "Could be a very good post player", 
        },
        "Good": {
            "Could be a good post player",
        },
        "Average": {"Above average post moves"},

        "Bad": ""
    }
    OutsideShooting = {
        "Great" : {
            "Could be an excellent shooter"
        },
        "Good": {
            "Could be a good shooter",
        },
        "Average": {"Could be an above average shooter"},

        "Bad": {"Does not have much of a shooting touch"}
    }
    Range = {"Average": {"Could really be a long-range shooter"}}

    Rebounding = {
        "Good": {"Can be a monster on the boards"},
        "Average": {"Can be a decent rebounder"},
        "Bad": {"Can never be a good rebounder"}
    }

    PlusDefense = {
        "Good": {"Can be a great all-around defensive player"},
        "Average": {"Can be a good all-around defensive player"},
        "Bad": {"Will always be a poor defender"}
    }

    InsideDefense = {
        "Bad": {"Will never be a good interior defender",}
    }
    PerimeterDefense = {
        "Bad": {"Will never be a good perimeter defender",}
    }
    IQ = {
        "Good": {"Can be a really smart player"},
        "Average": {
                "Can be a smart player"},
        "Bad": {
            "Can be prone to a lot of mental mistakes",
        }
    }
    Passing = {
        "Good": {"Can be a skilled passer with exceptional court vision"},
        "Average": {"Can be a skilled passer"}
    }
    Handling = {
        "Good": {"Can be a really good ball handler"},
        "Average": {"Can be a decent ball handler"},
        "Bad": {"Will be a below average ball handler"}
    }
    Speed = {
        "Great": {"Will be quick as lightning"},
        "Good": {
            "Will be very quick",
        },
        "Average": {"Can be a speedy player", },
        "Bad": {"Will always be a bit sluggish"}
    }

    All_Around = {
        "Good": {"Excellent all-around player"},
        "Average": {"Good all-around player"}
    }

    # === Mappings

    phrase_to_score = {}
    phrase_to_trait = {}

    def add_phrases(skill_dict, trait_name):
        for level, phrases in skill_dict.items():
            score = {"Bad": -1, "Average": 1, "Good": 2, "Great": 3, "Elite": 4}[level]
            for phrase in phrases:
                phrase_to_score[phrase] = score
                phrase_to_trait[phrase] = trait_name

    # Register all skills with explicit trait names
    add_phrases(InsideShooting, "IS_Eval")
    add_phrases(OutsideShooting, "OS_Eval")
    add_phrases(Range, "RNG_Eval")
    add_phrases(Rebounding, "Reb_Eval")
    add_phrases(PlusDefense, "Def_Eval")
    add_phrases(InsideDefense, "ID_Eval")
    add_phrases(PerimeterDefense, "PD_Eval")
    add_phrases(IQ, "IQ_Eval")
    add_phrases(Passing, "PA_Eval")
    add_phrases(Handling, "HND_Eval")
    add_phrases(Speed, "Spd_Eval")
    add_phrases(All_Around, "All_Around_Eval")

    # === Evaluation Function

    def evaluate_recruit(rec_eval):
        # Initialize all traits to 0
        trait_scores = {
            "IS_Eval": 0, "OS_Eval": 0, "RNG_Eval": 0, "Reb_Eval": 0,
            "Def_Eval": 0, "ID_Eval": 0, "PD_Eval": 0, "IQ_Eval": 0,
            "PA_Eval": 0, "HND_Eval": 0, "Spd_Eval": 0, "All_Around_Eval": 0
        }

        # Split and clean
        trait_sentences = rec_eval.replace("Recruiting Evaluation:", "").split(".")
        trait_sentences = [s.strip() for s in trait_sentences if s.strip() and not s.startswith("Projected Height")][0:-1]
        
        for sentence in trait_sentences:
            score = phrase_to_score.get(sentence)
            trait = phrase_to_trait.get(sentence)
            
            if score is not None and trait is not None:
                trait_scores[trait] = score
            
            #For the never be good perimeter and interior defenders
            if trait_scores["PD_Eval"] == 1:
                trait_scores["ID_Eval"] = -1
            
            if trait_scores["ID_Eval"] == 1:
                trait_scores["PD_Eval"] = -1



        

        return trait_scores

    scores = evaluate_recruit(rec_eval)
   
    return rec_eval, scores