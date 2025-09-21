from bs4 import BeautifulSoup
import requests
import re


#Given URL of player, it gets the measureables (Height, Weight, Wingspan, Vertical) and their skills and puts it into a dic

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



def transform_player_data(input_data):
    # Define the mapping from input keys to required keys
    key_mapping = {
        'Age': None,  # Not used in the output
        'Name': "Name",
        'InsideShot': 'IS',
        'BasketballIQ': 'IQ',
        'OutsideShot': 'OS',
        'Passing': 'Pass',
        'ShootingRange': 'Rng',
        'BallHandling': 'Hnd',
        #'Weight': 'Weight',
        'Finishing': 'Fin',
        'Driving': 'Drv',
        "Rebounding": 'Reb',
        'Strength': 'Str',
        'InteriorDefense': 'IDef',
        'Speed': 'Spd',
        'PerimeterDefense': 'PDef',
        'Stamina': 'Sta',
        'Weight': 'Weight',
        "Wingspan_inches": "Wingspan_inches",
        "Height_inches" : "Height_inches",
        "Vertical_float": "Vertical_float"
    }
    # Initialize the output dictionary
    output_data = {}
    
    # Apply the mapping to transform the input dictionary
    for key, new_key in key_mapping.items():
        if new_key is not None and key in input_data:
            output_data[new_key] = input_data[key]

    output_data["Name"] = output_data["Name"].replace("\n", "")

    #For matching with BPM dic
    output_data["height"] = output_data["Height_inches"]
    output_data["wingspan"] = output_data["Wingspan_inches"] 
    output_data["weight"] = output_data['Weight']
    output_data["vertical"] = output_data["Vertical_float"]

    del output_data["Height_inches"]
    del output_data["Wingspan_inches"]
    del output_data["Weight"]
    del output_data["Vertical_float"]



    
    return output_data



from bs4 import BeautifulSoup
import os




# For Flask App
def flask_get_player_info(player_html_link, from_file=False):
    
    if from_file:
        soup = BeautifulSoup(player_html_link, "html.parser")
        
    else:
        # fetch HTML from URL here
        response = requests.get(player_html_link)
        soup = BeautifulSoup(response.text, "html.parser")

    player_info = {}

    # Extract all table rows
    fullinfoList = soup.find("table").find_all("tr")

    playerID_search = fullinfoList[0].text.replace("[", " ").replace("]", " ").split(" ")
    #Search for Player ID (text starts wtih a "#")

    for i, text in enumerate(playerID_search):

        if text and "#" == text[0]:
            playerID = int(text[1:])
            break
    

    
    start_index = end_index = None
    for i in range(len(fullinfoList)):
        if "Age" in fullinfoList[i].text:
            start_index = i
        elif "Perimeter Defense" in fullinfoList[i].text:
            end_index = i + 1

    infoList = fullinfoList[start_index:end_index]
    player_info = {}

    for i in range(len(infoList)):
        text = infoList[i].text.strip().replace(" ", "")
        if i == 1:
            find_Outside = text.find("O")
            separated = split_number_and_letter(text[find_Outside:])
        else:
            separated = split_number_and_letter(text)

        if separated:
            if 'Height' in separated[0]:
                separated = separated[-4:]
            if 'Wingspan' in separated[0]:
                separated = separated[-4:]
            if 'Vertical' in separated[0]:
                separated = separated[-4:]

            if 'Weight' in separated[0]:
                player_info['Weight'] = float(f"{separated[1]}")
                start_idx = 2
                separated[start_idx] = separated[start_idx].replace("lbs.","")
            else:
                start_idx = 0

            for j in range(start_idx, len(separated), 2):
                key = separated[j].replace(":", "")
                if j + 1 < len(separated):
                    value = separated[j + 1]
                    player_info[key] = int(value)

    # Additional conversions
    player_info["Height_inches"] = convert_to_inches(extract_length(split_number_and_letter(infoList[2].text.strip().replace(" ", ""))))
    player_info["Wingspan_inches"] = convert_to_inches(extract_length(split_number_and_letter(infoList[4].text.strip().replace(" ", ""))))
    player_info["Vertical_float"] = Vert_convert_to_inches(extract_length(split_number_and_letter(infoList[5].text.strip().replace(" ", ""))))

    # Get Player Name
    name_soup = soup.find("h1")
    if name_soup:
        player_info["Name"] = name_soup.text.strip()
    else:
        player_info["Name"] = "Unknown"

    
    

    return transform_player_data(player_info) , playerID
'''
# --- Example usage ---
html_file = 'scripts\Hardwood - Johnnie Ray Player Profile.html'  # Path to the HTML you manually saved after CAPTCHA
if os.path.exists(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    player_data = get_player_info(html_content)
    print(player_data)
else:
    print(f"HTML file not found: {html_file}")

'''

'''
player_data = get_player_info(241946)
print("Player Data:", player_data)
'''