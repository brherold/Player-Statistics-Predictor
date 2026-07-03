from bs4 import BeautifulSoup
import requests
import re
import sys
import os





#Given URL of player, it gets the measureables (Height, Weight, Wingspan, Vertical) and their skills and puts it into a hash

def split_number_and_letter(s):
    s = s.replace('↑', '')
    s = s.replace('↓','')
    # Split the string into parts that contain numbers, fractions, letters, single quotes, and double quotes
    return re.findall(r'\d+|[¼½]|[^\d¼½\'"]+|[\'"]', s)

def extract_length(array):
    length = ""
    lengthArr = array[1:-4]
    for i in lengthArr:
        length += i
    
    return length

# For Height and Wingspan
def convert_to_inches(height):
    parts = height.split("'")
    if len(parts) == 2:
        feet = int(parts[0])
        inches_str = parts[1].rstrip('"').strip()  # Removing leading/trailing whitespace
        if inches_str:  # Check if inches part is not empty
            if '½' in inches_str:
                inches_str = inches_str.replace('½', '.5')
            elif '¼' in inches_str:
                inches_str = inches_str.replace('¼', '.25')
            inches = float(inches_str)
        else:
            inches = 0
    elif len(parts) == 1:
        feet = int(parts[0])
        inches = 0
    else:
        return None
    return (feet * 12) + inches

# For Vertical
def Vert_convert_to_inches(height):
    inches_str = height.rstrip('"')  # Removing the inch symbol
    if '½' in inches_str:
        inches_str = inches_str.replace('½', '.5')
    elif '¼' in inches_str:
        inches_str = inches_str.replace('¼', '.25')
    return float(inches_str)


# /H Page
# PlayersInputted File Format -> playerID-H.html (for predicited measureables) or playerID-D.html (for predicted skills)
def get_player_info_measureable(playerID):
    
    try:
        #playerID = file_path.split('-')
        with open(f'PlayersInputted/{playerID}-H.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as e:
            suffixes = ["HSFR", "HSSO", "HSJR", "HSSR"]
            folder = "C:/Users/branh/Documents/Hardwood PROJECTSSSSSS/Hardwood Recruit Searcher-EPM/PlayersHTML"
            for suffix in suffixes:
                file_path = os.path.join(folder, f"{playerID}-{suffix}-H.html")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        html_content = file.read()
                        #print(html_content)
                        #print(f"Loaded from {file_path}")
                    break
    if html_content is None:
        print(f"No file found for {playerID}")
        return None

    #print("QUAG")
    soup = BeautifulSoup(html_content, "html.parser")

    #_curr measureables are the current measureables of the highschool player we are given to then predict measureables after
    player_info = {}
    fullinfoList = soup.find("table").find_all("tr")
    

    #'''
    
    table_start = 3 if "College" not in fullinfoList[2].text else 4
    
    infoList = fullinfoList[table_start:]
    #'''
    #Gets Current (Fully Grown) Measureables
    for i, info in enumerate(infoList):
        #3 HT, 4 WT, 5 Wing, 6 Vert, 12 RecEval (for getting predicted height)
        text = info.text.strip().replace(" ", "")
        separated = split_number_and_letter(text)
        #print(separated)
        if separated:
            if 'Height' in text:
                separated = separated[-4:]
            if 'Wingspan' in text:
                separated = separated[-4:]
            if 'Vertical' in text:
                separated = separated[-4:]
            
            # Handle special cases
            if 'Weight' in text:
                player_info['weight_curr'] = float(f"{separated[1]}")
                
                start_idx = 2
                separated[start_idx] = separated[start_idx].replace("lbs.","")



    #'''
    #print(player["PlayerID"], split_number_and_letter(infoList[3].text.strip().replace(" ", "")))
    player_info["height_curr"] = convert_to_inches(extract_length(split_number_and_letter(infoList[3].text.strip().replace(" ", ""))))
    player_info["wingspan_curr"] = convert_to_inches(extract_length(split_number_and_letter(infoList[5].text.strip().replace(" ", ""))))
    player_info["vertical_curr"] = Vert_convert_to_inches(extract_length(split_number_and_letter(infoList[6].text.strip().replace(" ", ""))))
    
   

    #Get Freshman(and other) Height and Weight
    soup = BeautifulSoup(html_content, "html.parser")

    player_class = ""

    try:
        table = soup.find("table", class_="stats-table-medium_font")
        rows = table.find_all('tr')

        #Gets Freshman Height and wingpspan
        #print(row)
        height = convert_to_inches((rows[1].find_all('td')[33]).text)

        # For getting Freshman Weight
        weight_cell = rows[1].find_all('td')[34]     
        weight = int(weight_cell.get_text(strip=True))

        
        player_info[f"height_1"] = height
        player_info[f"weight_1"] = weight
    except:
        #For Internationals
        player_class = "INT"
        player_info[f"height_1"] = None
        player_info[f"weight_1"] = None

    pageList = soup.find("th").find_all("a")[1:]

    player_name = pageList[0].find("img")["title"].replace("Bookmark ","")

    #playerID = int(pageList[-1].get("href").split("/")[-1])

    player_info["Name"] = player_name
    player_info["Player_ID"] = playerID

    if player_class !=  "INT":
        #Find Class of Player
        class_check = fullinfoList[1].text
        #print(class_check)
        if "High School Freshman" in class_check:
            player_class = "HSFR"
        elif "High School Sophomore" in class_check:
            player_class = "HSSO"
        elif "High School Junior" in class_check:
            player_class = "HSJR"
        elif "High School Senior" in class_check:
            player_class = "HSSR"
        elif "Redshirt Freshman" in class_check:
            player_class = "Col-2"
        elif "Redshirt" not in class_check and "Sophomore" in class_check:
            player_class = "Col-2"
        elif "Freshman" in class_check:
            player_class = "Col-1"
        else:
            #Use current measurables 
            player_class = "OLDER"


    
    player_info["Class"] = player_class
    



    return player_info


'''
player_data = get_player_info_measureable(241946)
print("Player Data:", player_data)

'''

