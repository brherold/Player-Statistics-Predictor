from bs4 import BeautifulSoup
import requests
import os
from RecruitSkillPredictorCol.scripts.getPred_4 import *

def save_html_to_file(content_bytes, filename):
    # Write raw bytes—use wb to overwrite or ab to append
    with open(filename, 'wb') as f:
        f.write(content_bytes)

hardwood_link = "http://onlinecollegebasketball.org/player/"


#
def update_player_html(folder_path, folder_name):
    count = 1
    # List all files and directories in the given folder path
    for filename in os.listdir(folder_path):
        # Create the full file path by joining the folder path and filename
        file_path = os.path.join(folder_path, filename)
        player_code = filename.split(".")[0]

        player_link = hardwood_link + str(player_code)

        response = requests.get(player_link + "/D")
        html_content = response.content
        os.makedirs(folder_name, exist_ok=True)  # Create folder if it doesn't exist
        save_html_to_file(html_content, f"{folder_name}/{filename}")
        print(count)
        count += 1

'''
# Example usage:
folder_path = 'RecruitSkillPredictorCol/Players-Checked-HTML'
folder_name = 'RecruitSkillPredictorCol/Players-Checked-HTML-Updated'
update_player_html(folder_path, folder_name)
'''
def update_player_html():
    folder_path = 'RecruitSkillPredictorCol/Players-Checked-HTML'
    count = 1
    # List all files and directories in the given folder path
    for filename in os.listdir(folder_path):
        # Create the full file path by joining the folder path and filename
        file_path = os.path.join(folder_path, filename)
        player_code = filename.split(".")[0]


        for i in range(1,6):
            try:
                getPredictedSkills(player_code,i)
            except:
                print(f"Error: {player_code}, year: {i}")
        count += 1
        
        print(count)

update_player_html()

