height_inc_dec = .1131
weight_inc_dec = .3959
wingspan_inc_dec = .1131
vertical_inc_dec = .2451
import math

def height_pred(fresh_height):
  res = fresh_height + fresh_height * height_inc_dec
  #print(res)
  res_string = str(res)

  dec_values = res_string.split(".")[-1]
  tenths = int(dec_values[0])

  if tenths < 4:
    return math.floor(res)  #Round down .00
  if tenths >= 4 and tenths < 9:
    if tenths < 8:
      return round(res * 2) / 2 #rounds to .5
    else:
      return math.floor(res * 2) / 2 #rounds to .5
  else:
    return math.ceil(res)  # Round up 1.00
  
def weight_pred(fresh_weight):
  res = fresh_weight + fresh_weight * weight_inc_dec
  return round(res / 5) * 5  #rounds to 0 or 5

######

#F_W freshWeight
#C_W currentWeight

def playerGrowthPercent(C_W,F_W):
    return ((C_W - F_W)/F_W) / weight_inc_dec


#P = playerGrowthPercent(C_W,F_W,weight_inc_dec)


#C -> Current value
#G -> Full growth rate of that measureable (ex: .1101 = wingspan)
def predicted(C,G,P):
    if P >= 1: #player done growing
        return C
    F = C/ (1 + P * G)

    return F * (1 + G)



def wingspan_pred(current_wingspan,P):
   pred_wingspan = round(predicted(current_wingspan,wingspan_inc_dec,P)*4)/ 4
   if pred_wingspan >= 96:
      return 95.5
   elif pred_wingspan < current_wingspan:
      return current_wingspan
   return pred_wingspan


def vertical_pred(current_vertical,P):
  pred_vertical = round(predicted(current_vertical,vertical_inc_dec,P)*2)/ 2
  if pred_vertical < current_vertical:
    return current_vertical
  return pred_vertical

