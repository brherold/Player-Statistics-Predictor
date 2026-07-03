#'''
from  scripts.recruitMeasureablePredictorWeight import *
from scripts.getPlayerInfoMeasureable import *
from scripts.predict_measurable import *
#'''
'''
from  recruitMeasureablePredictorWeight import *
from getPlayerInfoMeasureable import *
from predict_measurable import *
'''





def getPredictedMeasureables(playerID):

    player_info = get_player_info_measureable(playerID)

    player_measureable_dic = {}

    #Use Current Measurables for Internationals or Players whose Year is  >= 3
    if player_info["Class"] == "OLDER":

            
        player_measureable_dic["Name"] = player_info["Name"]
        player_measureable_dic["playerCode"] = player_info["Player_ID"]
        player_measureable_dic["height"] = player_info["height_curr"]
        player_measureable_dic["weight"] = player_info["weight_curr"]
        player_measureable_dic["wingspan"] = player_info["wingspan_curr"]
        player_measureable_dic["vertical"] = player_info["vertical_curr"]

        return player_measureable_dic

    
    #Predict HS Measurables
    else:
        
        measurable_pred = predict_measurables(player_info["Class"], player_info["height_curr"], player_info["weight_curr"]
                                              , player_info["wingspan_curr"], player_info["vertical_curr"], player_info["height_1"]
                                              , player_info["weight_1"], player_info["height_projected"])
        
        pred_height = measurable_pred["height_end"]
        pred_weight = measurable_pred["weight_end"]
        pred_wingspan = measurable_pred["wingspan_end"]
        pred_vertical = measurable_pred["vertical_end"]

    
    player_info["Name"]    
    player_measureable_dic["Name"] = player_info["Name"]
    player_measureable_dic["playerCode"] = player_info["Player_ID"]
    player_measureable_dic["height"] = pred_height
    player_measureable_dic["weight"] = pred_weight
    player_measureable_dic["wingspan"] = pred_wingspan
    player_measureable_dic["vertical"] = pred_vertical

    

    
    return player_measureable_dic




#print(getPredictedMeasureables(232458))