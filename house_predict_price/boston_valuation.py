from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston_dataset = pd.read_csv('./BostonHousing.csv')
data = boston_dataset.drop(['medv'], axis=1)
target_b = boston_dataset['medv']
#data['price'] = boston_dataset_target
features = data.drop(["indus", "age"], axis=1)

log_prices = np.log(target_b)
target = pd.DataFrame(data=log_prices.values, columns=['price']) 

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

# property_stats = np.ndarray(shape = (1,11))
# property_stats[0][CRIME_IDX] = features['crim'].mean()
# property_stats[0][ZN_IDX] = features['zn'].mean()
# property_stats[0][CHAS_IDX] = features['chas'].mean()

property_stats = features.mean().values.reshape(1,11) #масив з середніми по всіх стовпцях

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    hight_confidence=True):
    
    #configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
   
    #Calc Range
    if hight_confidence:
        upper_bound = log_estimate+2*RMSE
        lower_bound = log_estimate-2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate+RMSE
        lower_bound = log_estimate-RMSE
        interval = 68
        
    return log_estimate, upper_bound, lower_bound, interval

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE/np.median(target_b)

log_est, upper, lower, conf = get_log_estimate(9, students_per_classroom=15, next_to_river=False, hight_confidence=False)

#Convert to today's dollars
dollar_est = np.e**log_est*1000*SCALE_FACTOR
dollar_hi = np.e**upper*1000*SCALE_FACTOR
dollar_low = np.e**lower*1000*SCALE_FACTOR


rounded_est = np.round(dollar_est,-3)
rounded_hi = np.round(dollar_hi,-3)
rounded_low = np.round(dollar_low,-3)

print(f'The estimated property value is {rounded_est}.')
print(f'At {conf}% confidence the valuation range is {rounded_est}.\n USD {rounded_low} at the lower end to USD {rounded_hi} at hight end.')


def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """
        This function makes magic
    """

    if rm < 1 or ptratio < 1 or ptratio >= 100:
        print('That is unrealistic. Try again')
        return
    
    log_est, upper, lower, conf = get_log_estimate(rm, students_per_classroom=ptratio, next_to_river=chas, hight_confidence=large_range)

    #Convert to today's dollars
    dollar_est = np.e**log_est*1000*SCALE_FACTOR
    dollar_hi = np.e**upper*1000*SCALE_FACTOR
    dollar_low = np.e**lower*1000*SCALE_FACTOR


    rounded_est = np.round(dollar_est,-3)
    rounded_hi = np.round(dollar_hi,-3)
    rounded_low = np.round(dollar_low,-3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is {rounded_est}.\n USD {rounded_low} at the lower end to USD {rounded_hi} at hight end.')
