import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

def predict_measurables(year, height_curr, weight_curr, wingspan_curr, vertical_curr, height_1, weight_1,height_projected):
    """
    Predict end-of-season measurables for a player using pre-trained Ridge models
    and round predictions to specified increments.
    """
    
    if "INT" in year:
        features = [height_curr, weight_curr, wingspan_curr, vertical_curr, height_projected]
        
    else:
        features = [height_curr, weight_curr, wingspan_curr, vertical_curr, height_1, weight_1]

    input_array = np.array(features).reshape(1, -1)
    
    results = []
    for meas in ['Height', 'Weight', 'Wingspan', 'Vertical']:
        model, model_features, df_train = joblib.load(f'MeasurableModel/{year}_{meas}.pkl')
        pred = model.predict(input_array)[0]
        results.append(pred)
    
    # Convert to NumPy array for rounding
    y_pred = np.array(results).reshape(1, -1)
    
    # Round predictions to desired increments
    y_pred[:, 0] = max(float(np.round(y_pred[:, 0] * 2) / 2), height_curr)      # height to nearest 0.5
    y_pred[:, 1] = max(float(np.round(y_pred[:, 1] / 5) * 5), weight_curr)       # weight to nearest 5
    y_pred[:, 2] = max(float(np.round(y_pred[:, 2] / 0.25) * 0.25), wingspan_curr) # wingspan to nearest 0.25
    y_pred[:, 3] = max(float(np.round(y_pred[:, 3] * 2) / 2), vertical_curr)   # vertical to nearest 0.5
    
    return {
        'height_end': float(y_pred[0, 0]),
        'weight_end': float(y_pred[0, 1]),
        'wingspan_end': float(y_pred[0, 2]),
        'vertical_end': float(y_pred[0, 3])
    }
'''
# Example usage
preds = predict_measurables(
    year="HSSO",
    height_curr=74,
    weight_curr=190,
    wingspan_curr=81,
    vertical_curr=36,
    height_1=74,
    weight_1=180
)
print(preds)
'''