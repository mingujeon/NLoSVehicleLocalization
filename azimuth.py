import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path = 'cls_features/SA/Wall_paint_L/Color_Left_Test/Site1_L_10_Trial9.csv' # sound feature path

df = pd.read_csv(path)

print(df.loc[0][0])
# angles = np.deg2rad(np.arange(-90, 91, 1))

name = path.split('/')[-2] 
if not os.path.isdir('azimuth/'+name) :
    os.mkdir('azimuth/'+name)


for i in range(len(df)) :
    #azi = []
    azi = df.iloc[i, :180].tolist()
    #for j in range(180) :
    #    azi.append(df.loc[i][j])
    # print(len(azi))
    # print(azi)

    #azi[0] = 1
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    angles = np.deg2rad(np.linspace(0, 179, 180))
    # Plot the intensities
    #print(len(angles))
    #print(len(azi))

    ax.plot(angles, azi)

    # Set the direction to clockwise and 0 degrees to the top
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)


    # Set the labels for the angles
    ax.set_xticks(np.deg2rad(np.arange(0, 181, 30)))  # Labels at every 30 degrees
    ax.set_xticklabels([-90, -60, -30, -0, 30, 60, 90])

    ax.set_yticks([0.0, 0.1,0.2,0.3,0.4,0.5]) 
    ax.set_yticklabels([0.0, 0.1,0.2,0.3,0.4,0.5])

    # Set the title and show the plot
    ax.set_title(str(round(df.loc[i][180],2)) + ' s')
    plt.savefig('azimuth/'+name+'/azimuth_' + str(round(df.loc[i][180],2))+'.png')
    # plt.show()
