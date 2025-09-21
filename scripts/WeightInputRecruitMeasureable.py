from  scripts.recruitMeasureablePredictorWeight import *
from scripts.getPlayerInfoMeasureable import *

#from  recruitMeasureablePredictorWeight import *
#from getPlayerInfoMeasureable import *

from bs4 import BeautifulSoup
import requests

#Find player coefficient of Bigs (uses EPM coefficients to determine value)
def player_coefficient(height,weight,wingspan,vertical):
    # Given the average player's measurements

    #weight scaled by every 10 lbs
    weight = weight / 10

    # Initial coefficients 
    height_coef = 0.161699
    weight_coef = 0.112239
    wingspan_coef = 0.103767
    vertical_coef = 0.122199

    # Calculate the weighted sum for the average player
    player_value = (height * height_coef) + (weight * weight_coef) + (wingspan * wingspan_coef) + (vertical * vertical_coef)
    average_player_value = 28.086244000000004

    player_coeff = round(player_value - average_player_value,2)
  
    #return player_coeff
    grade = 50 + 20*player_coeff 

    return player_coeff, grade 



def getPredictedMeasureables(playerCode):

    player_info = get_player_info_measureable(playerCode)

    player_measureable_dic = {}

    if not player_info["Fresh_Height"]:

        player_measureable_dic["Name"] = player_info["Name"]
        player_measureable_dic["playerCode"] = player_info["Player_ID"]
        player_measureable_dic["height"] = player_info["Height"]
        player_measureable_dic["weight"] = player_info["Weight"]
        player_measureable_dic["wingspan"] = player_info["Wingspan"]
        player_measureable_dic["vertical"] = player_info["Vertical"]

        return player_measureable_dic

        
    P = playerGrowthPercent(player_info["Weight"],player_info["Fresh_Weight"])
    
    pred_height = height_pred(player_info["Fresh_Height"])
    if pred_height < player_info["Height"]:
        pred_height = player_info["Height"]

    pred_weight = weight_pred(player_info["Fresh_Weight"])
    if pred_weight < player_info["Weight"]:
        pred_weight = player_info["Weight"]

    pred_wingspan = wingspan_pred(player_info["Wingspan"],P)
    pred_vertical = vertical_pred(player_info["Vertical"],P)


    
    player_measureable_dic["Name"] = player_info["Name"]
    player_measureable_dic["playerCode"] = player_info["Player_ID"]
    player_measureable_dic["height"] = pred_height
    player_measureable_dic["weight"] = pred_weight
    player_measureable_dic["wingspan"] = pred_wingspan
    player_measureable_dic["vertical"] = pred_vertical

    
    return player_measureable_dic




#print(getPredictedMeasureables(237714))


def getPredictedMeasureables_asker():

    player_name = input("Name: ")
    print()

    fresh_height = float(input("Freshman Height (inches): "))
    curr_height = float(input("Current Height (inches): "))
    fresh_weight = float(input("Freshman Weight (lbs): "))
    curr_weight = float(input("Current Weight (lbs): "))
    curr_wingspan = float(input("Current Wingspan (inches): "))
    curr_vert = float(input("Current Vertical (inches): "))
    


    player_info = {"Name": player_name, "Fresh_Height": fresh_height, "Height": curr_height, "Fresh_Weight": fresh_weight, "Weight": curr_weight,  "Wingspan" : curr_wingspan, "Vertical": curr_vert}


    player_measureable_dic = {}


        
    P = playerGrowthPercent(player_info["Weight"],player_info["Fresh_Weight"])
    
    pred_height = height_pred(player_info["Fresh_Height"])
    if pred_height < player_info["Height"]:
        pred_height = player_info["Height"]

    pred_weight = weight_pred(player_info["Fresh_Weight"])
    if pred_weight < player_info["Weight"]:
        pred_weight = player_info["Weight"]

    pred_wingspan = wingspan_pred(player_info["Wingspan"],P)
    pred_vertical = vertical_pred(player_info["Vertical"],P)

    player_coeff, grade  = player_coefficient(pred_height,pred_weight,pred_wingspan,pred_vertical)

    
    player_measureable_dic["Name"] = player_info["Name"]
    player_measureable_dic["height"] = pred_height
    player_measureable_dic["weight"] = pred_weight
    player_measureable_dic["wingspan"] = pred_wingspan
    player_measureable_dic["vertical"] = pred_vertical
    player_measureable_dic["Grade"] = grade
    


    return player_measureable_dic


#python -m scripts.WeightInputRecruitMeasureable  
#'''
while(True):
  print(getPredictedMeasureables_asker())
  print()
  #'''