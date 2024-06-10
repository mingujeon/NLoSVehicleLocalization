import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

excel_file_path = 'gt/Site1_R.xlsx'
gt_df = pd.read_excel(excel_file_path)
gt = list(gt_df['ground_truth'])

for i in [10,25,50,100] :
    pkl_file_path = 'cls_features/SA/Site1_R/Site1_R_10_Trial6/avg_Tracking_' +str(i)+ '.pkl'
    with open(pkl_file_path, 'rb') as f:
        pred_data = pickle.load(f)

    total_pred_x = pred_data['x_position']

    x = min(len(gt_df), len(total_pred_x))

    rmse = mean_squared_error(gt[:x], list(total_pred_x)[:x], squared=False)

    print(str(i) + " Particle RMSE:", rmse)