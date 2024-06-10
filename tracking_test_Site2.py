from re import T
from ParticleFilter_Site2 import Map, ParticleFilter
from config.mapInfo import map_info

import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import math
import time

N = 20


def main(args):
    log = open("log.txt", 'a')
    print("file args.input: %s"%(args.input))
    log.write("file : %s\n"%(args.input))
    all_ASPLE, all_ASPLE_var, all_ASPLE_weight, all_ASPLE_avg = {}, {}, {}, {}
    file_name = args.input.split('/')[4]
    print('file: ', file_name)

    necessary_log = []
    weight_rate = [0,0.3,1,1]

    for N_PARTICLE in [10,25,50,100]:  
        try:
            os.system("rm -rf tmp")
        except:
            pass
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        if not os.path.isdir(str(args.input.split('/')[3]) + '_' + str(N_PARTICLE)):
            os.mkdir(str(args.input.split('/')[3]) + '_' + str(N_PARTICLE))
            print('Tracking image path: ', args.input.split('/')[3]) 


        _,env,*_ = args.input.split('/')

        features = pd.read_csv(args.input)
        i_max = 1
        for i, col in enumerate(features.columns):
            if 'feat' in col:
                i_max = i + 1 
        features, times = np.split(features.to_numpy(), [i_max], axis=1)
        features = np.array(features, dtype=np.float32)

        ###############################
        # Likelihood shaping function #
        ###############################
        features = np.power(features,4)
        features = features - np.quantile(features, 0.75, axis=1).reshape(-1,1)

        total_pred_x = []
        total_pred_avg_x = []
        total_pred_x_var = []
        total_pred_weight = []

        for n in range(N):
            _map = Map(map_info[env])
            pf = ParticleFilter(N_PARTICLE, _map, args.epsilon, weight_rate = weight_rate)

            f = open(args.input.replace(".csv", "_result.txt"), 'w')
            pred_t, pred_x, pred_x_var = [], [], []
            pred_weight, pred_avg_x = [], []
            start = time.time()
            for i in range(times.shape[0]):
                pred_weight.append([])
            
                if i != times.shape[0] - 1:
                    dt = times[i+1]-times[i]
                    pf.propagate(dt[0])

                if pf.time <= 0.7 * (len(times)/80): 
                    weight_rate = [0,0.3,1,1]
                elif pf.time <= 1.9 * (len(times)/80): 
                    weight_rate = [1,0.7,0.3,0.7]
                elif pf.time <= 5.5 * (len(times)/80): 
                    weight_rate = [1, 0.6, 0.2,0.7]
                elif pf.time <= 6.9 * (len(times)/80): 
                    weight_rate = [1,0.7,0.3,0.7]
                else :
                    weight_rate = [0,0.3,1,1]  


                pf.update(features[i], weight_rate)
                
                f.write("%f, %f, %f\n"%(pf.time, pf.Max_prediction.x, pf.prediction.y))
                pred_t.append(pf.time)
                pred_x.append(pf.Max_prediction.x)
                pred_avg_x.append(pf.prediction.x)

                
                sorted_particles = sorted(zip(pf.particles, pf.weight), key=lambda x: x[1], reverse=True)
                pred_weight[i] = [[p.x, p.y, w] for p, w in sorted_particles]

                pf.resample()

                # Variance check
                x_var = [particle.x for particle in pf.particles][:int((pf.n)*(1-pf.epsilon))] # only non-random sampled particles
                x_var = np.std(x_var)
                pred_x_var.append(x_var)
                
                if args.render :
                    pf.render(str(args.input.split('/')[3]) + '_' + str(N_PARTICLE), args.ray)
            
            if n == 0:
                print("%d particle execution time : %.2f ms" %(N_PARTICLE, (time.time()-start)*1000/times.shape[0]))
                log.write("%d_particle_execution_time(ms) : %.2f\n" %(N_PARTICLE, (time.time()-start)*1000/times.shape[0]))
                necessary_log.append("%d_particle_execution_time(ms) : %.2f\n" %(N_PARTICLE, (time.time()-start)*1000/times.shape[0]))
        
        total_pred_x.append(pred_x)
        total_pred_x_var.append(pred_x_var)
        total_pred_weight.append(pred_weight)
        total_pred_avg_x.append(pred_avg_x)
    
        columns = ['time_step_' + str(i) for i in range(len(total_pred_weight))]

        df_weights = pd.DataFrame(list(zip(*total_pred_weight)), columns=columns)

        save_path = args.input.replace(file_name, str(N_PARTICLE) +"_Particle_info.xlsx")

        df_weights.to_excel(save_path, index=False)

        # save result
        dt = dt[0]
        gt_data = pd.read_excel("gt/Site2_L_10_Trial8.xlsx") # gt path
        gt_times = gt_data['time'].values
        gt_x_positions = gt_data['ground_truth'].values
        gt = np.column_stack((gt_times, gt_x_positions))
        gt_grad = 0.1

        # gt = np.loadtxt(args.input.replace(file_name, "ground_truth.txt"), dtype=float, delimiter=",")
        total_pred_x = np.array(total_pred_x) # [:,int((gt[0,0]-2)/dt)+1:]
        total_pred_avg_x = np.array(total_pred_avg_x) # [:,int((gt[0,0]-2)/dt)+1:]
        total_pred_x_var = np.array(total_pred_x_var)
        end_time = (total_pred_x.shape[1]*dt-2)
        real_gt = {'time': [0, end_time], 'x_position': [gt[0,1],end_time*gt_grad+gt[0,1]]}
        
        # maximum DoA baseline
        start2 = time.time()
        doa = (np.argmax(features, axis=1)-90) * math.pi / 180
        doa_f = 12 * np.tan(doa)
        # doa_f = doa_f[int((gt[0,0]-2)/dt):]
        # doa_f = doa_f[:total_pred_x.shape[1]]

        # bound 
        doa_f = np.maximum(doa_f, -22)
        doa_f = np.minimum(doa_f, 22)

        doa_f = {'time': np.linspace(0, end_time, doa_f.shape[0]), 'x_position': doa_f}
        doa_f_df = pd.DataFrame(doa_f)
        excel_save_path = file_name+str(N_PARTICLE)+'_path_to_save_excel_file.xlsx'
        doa_f_df.to_excel(excel_save_path, index=False)
        print("DOA time : %f"%((time.time()-start2)/features.shape[0]*1000))

        # ASPLE
        ASPLE_df = pd.DataFrame(columns=['time','x_position'])
        for t in range(total_pred_x.shape[1]):
            for i in range(total_pred_x.shape[0]):
                ASPLE_df = ASPLE_df._append({'time':t*dt, 'x_position':total_pred_x[i,t]}, ignore_index=True)
        all_ASPLE[N_PARTICLE] = ASPLE_df

        # ASPLE_avg
        ASPLE_df_avg = pd.DataFrame(columns=['time','x_position'])
        for t in range(total_pred_avg_x.shape[1]):
            for i in range(total_pred_avg_x.shape[0]):
                ASPLE_df_avg = ASPLE_df_avg._append({'time':t*dt, 'x_position':total_pred_avg_x[i,t]}, ignore_index=True)
        all_ASPLE_avg[N_PARTICLE] = ASPLE_df_avg


        # ASPLE variance
        ASPLE_df_var = pd.DataFrame(columns=['time','x_var'])
        for t in range(total_pred_x_var.shape[1]):
            for i in range(total_pred_x_var.shape[0]):
                ASPLE_df_var = ASPLE_df_var._append({'time':t*dt, 'x_var':total_pred_x_var[i,t]}, ignore_index=True)
        all_ASPLE_var[N_PARTICLE] = ASPLE_df_var

        with plt.style.context(("seaborn-paper",)):
            sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
            sns.lineplot(data=ASPLE_df, x='time', y='x_position',color='r', label='Particle filter prediction')
            plt.xlabel("time [sec]")
            plt.ylabel("x position [m]")
            plt.legend()
            plt.savefig(args.input.replace(file_name, "%d_%d.png"%(N_PARTICLE, N)))
            plt.close()

        # avg 
        with plt.style.context(("seaborn-paper",)):
            # sns.lineplot(data=real_gt, x='time', y='x_position', color='k', label='Ground truth')
            # sns.lineplot(data=regress_gt, x='time', y='x_position', color='k', linestyle="dashed", label='Ground truth regression')
            sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
            sns.lineplot(data=ASPLE_df_avg, x='time', y='x_position', color='r', label='Particle filter prediction')
            plt.xlabel("time [sec]")
            plt.ylabel("x position [m]")
            plt.legend()
            # plt.savefig(args.input.replace("out_multi.csv", "%d_%d.png"%(N_PARTICLE, N)))
            plt.savefig(args.input.replace(file_name, "avg_%d_%d.png"%(N_PARTICLE, N)))
            plt.close()


        # Variance
        with plt.style.context(("seaborn-paper",)):
            sns.lineplot(data=ASPLE_df_var, x='time', y='x_var', color='r', label='Particle filter prediction')
            plt.xlabel("time [sec]")
            plt.ylabel("particle std [m]")
            plt.savefig(args.input.replace(file_name, "variance_%d_%d.png"%(N_PARTICLE, N)))
            plt.close()


    # final all    
    with plt.style.context(("seaborn-paper",)):

        sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
        for n, ASPLE_df in all_ASPLE.items():
            sns.lineplot(data=ASPLE_df, x='time', y='x_position', label='%d max_particles'%n)
            ASPLE_df.to_pickle(args.input.replace(file_name, "Tracking_%d.pkl"%n))
        plt.xlabel("time [sec]")
        plt.ylabel("x position [m]")
        plt.legend()
        name = args.input.split('/')
        plt.title(name[1] + " " + name[2])
        # plt.savefig(args.input.replace("out_multi.csv", "tracking.png"))
        plt.savefig(args.input.replace(file_name, "tracking.png"))
        plt.close()

    # avg 
    with plt.style.context(("seaborn-paper",)):
        sns.lineplot(data=doa_f, x='time', y='x_position', color='b', label='Max DoA prediction')
        for n, ASPLE_df in all_ASPLE.items():
            sns.lineplot(data=ASPLE_df, x='time', y='x_position', label='%d max_particles'%n)
        for n, ASPLE_df_avg in all_ASPLE_avg.items():
            sns.lineplot(data=ASPLE_df_avg, x='time', y='x_position', label='%d avg_particles'%n)
            ASPLE_df_avg.to_pickle(args.input.replace(file_name, "avg_Tracking_%d.pkl"%n))
        plt.xlabel("time [sec]")
        plt.ylabel("x position [m]")
        plt.legend()
        name = args.input.split('/')
        plt.title(name[1] + " " + name[2])
        # plt.savefig(args.input.replace("out_multi.csv", "tracking.png"))
        plt.savefig(args.input.replace(file_name, "avg_tracking.png"))
        plt.close()

    # variance
    with plt.style.context(("seaborn-paper",)):
        for n, ASPLE_df_var in all_ASPLE_var.items():
            sns.lineplot(data=ASPLE_df_var, x='time', y='x_var', label='%d particles'%n)

            ASPLE_df_var.to_pickle(args.input.replace(file_name, "%d.pkl"%n))
        plt.xlabel("time [sec]")
        plt.ylabel("particle std [m]")
        plt.legend()
        name = args.input.split('/')
        plt.title(name[1] + " " + name[2])
        plt.savefig(args.input.replace(file_name, "variance.png"))
        plt.close()
    

def str_to_float_list(s):
    return [float(item) for item in s.split()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ParticleFilter")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--weight_rate', type=str_to_float_list, required=False)
    args = parser.parse_args()
    main(args)