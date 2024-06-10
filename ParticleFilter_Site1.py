import numpy as np
import random
import math
import os
from copy import deepcopy
from shapely.geometry import LineString, Point
from shapely.affinity import scale
from itertools import permutations, combinations

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle


def mirror_point(point, line, max_scale=100):
    """
    mirroring point w.r.t line segment
    """
    line_long = scale(line, xfact=max_scale, yfact=max_scale)
    p_inter = line_long.interpolate(line_long.project(point))
    p_new = Point(2 * p_inter.x - point.x , 2 * p_inter.y - point.y)

    return p_new

def measurement(z, angle, n):
    idx = (angle/math.pi+0.5)*n
    idx_u = min(math.floor(idx+0.5), n-1)
    idx_l = max(math.floor(idx-0.5), 0)

    z_inter = z[idx_u] * (idx-idx_l-0.5) + z[idx_l] * (idx_u+0.5-idx)

    return z_inter

class Particle:
    def __init__(self, weight_rate):
        self.weight = weight_rate
        self.w_direct = weight_rate[0]
        self.w_reflect_1st = weight_rate[1]
        self.w_reflect_2nd = weight_rate[2]
        self.w_diffract = weight_rate[3]
        # print('weight_rate: ', weight_rate)


    def change_weight_rate(self, weight_rate) :
        self.weight = weight_rate
        self.w_direct, self.w_reflect_1st, self.w_reflect_2nd, self.w_diffract = self.weight


    def random_initialize(self, _map):
        """
        Randomly initialize particle state
        Depend on map(only initialize at unknown region)
        """
        self.x = random.uniform(-_map.road_range[0]/2, _map.road_range[0]/2)
        self.y = random.uniform(0, _map.road_range[1]) + _map.road_range[2]

        self.head = random.choice([0, math.pi])
        # self.head = random.uniform(0,2*math.pi)
        self.v = random.random()*3 # 0~10m/s
        self.w_head = self.v/2.8*math.tan(math.radians((random.random()-0.5)*10))

    
    def set_weight(self):
        self.w_direct, self.w_reflect_1st, self.w_reflect_2nd, self.w_diffract = self.weight


    def move(self, t):
        """
        Move given time step
        Langevin's model
        Noise level 
        """
        self.x = self.x + self.v * math.cos(self.head) * t
        self.y = self.y + self.v * math.sin(self.head) * t
        self.head = self.head + self.w_head * t 
        self.v = self.v + random.normalvariate(0,1) 


    def measure_prob_addition(self, _map, z, weight_rate):
        """
        Measuring Pseudo Likelihood
        """
        self.change_weight_rate(weight_rate)
        # print(self.weight)
        p = Point(self.x, self.y)
        # particle go out of map
        if self.x > _map.width/2 or self.x < -_map.width/2 or self.y > _map.height or self.y < _map.road_range[2] + 0.2 :
            return 0

        direct = _map.direct(p)
 
        w_direct, w_reflect_1st, w_reflect_2nd, w_diffract = self.w_direct, self.w_reflect_1st, self.w_reflect_2nd, self.w_diffract
        
        prob_list = []

        # direct
        if len(direct) > 0:
            prob_list += [measurement(z, direct[1], z.shape[0]) * w_direct]

            reflect_1st = _map.reflect_1st(p)
            for r in reflect_1st:
                prob_list += [measurement(z, r[1], z.shape[0]) * w_reflect_1st]
        else:
            reflect_1st = _map.reflect_1st(p)
            for r in reflect_1st:
                prob_list += [measurement(z, r[1], z.shape[0]) * w_reflect_1st]

            reflect_2nd = _map.reflect_2nd(p)
            for r in reflect_2nd:
                prob_list += [measurement(z, r[1], z.shape[0]) * w_reflect_2nd]

            diffract = _map.diffract(p)
            for di in diffract:
                prob_list += [measurement(z, di[1], z.shape[0]) * w_diffract]

        prob_list.sort(reverse=True)
        prob = np.sum(prob_list)
        
        return prob
    

class Map:
    """
    Design map
    """
    def __init__(self, map_info):
        self.width = map_info['width']
        self.height = map_info['height']
        self.walls  = map_info['walls'] 
        self.corners = map_info['corners'] 
        self.thres = map_info['diffraction_angle_threshold']
        self.road_range = map_info['road_range']
        self.front_range = map_info['front_range']
        self.front_height = map_info['front_height']
        self.ego_vehicle = Point(0,0)


    def move(self):
        """
        Only for dynamic situation
        """
        assert NotImplementedError

    def direct(self, p, walls=None, vis=False):
        """
        Direct acoustic ray direction
        """
        if walls is None:
            walls = self.walls
        l = LineString([p, self.ego_vehicle])

        for wall in walls:    
            if l.intersects(wall):
                return []
        
        r = math.sqrt((self.ego_vehicle.x-p.x)**2 + (self.ego_vehicle.y-p.y)**2)
        theta = math.atan(p.x/p.y)

        if not vis:
            return [r , theta]

        else:
            return [r, theta], [l]
        

    def direct_line(self, l, walls=None):
        if walls is None:
            walls = self.walls
        
        for wall in walls:
            if l.intersects(wall):
                return False

        return True
    

    def reflect_1st(self, p, vis=False):
        """
        1st-order reflected acoustic ray directions
        """
        reflects = []
        lines = []
        for wall in self.walls:
            p_prime = mirror_point(p, wall)
            l = LineString([self.ego_vehicle, p_prime])
            if l.intersects(wall):
                p_inter = l.intersection(wall)
                l1 = LineString([p, p_inter])
                l2 = LineString([p_inter, self.ego_vehicle])
                walls = [w for w in self.walls if w != wall]
                if self.direct_line(l1, walls):
                    loc = self.direct(p_inter, walls)
                    if len(loc) > 0:
                        reflects.append(loc)
                        lines += [l1,l2]

        if not vis:
            return reflects

        else:
            return reflects, lines
        

    def reflect_2nd(self,p, vis=False):
        """
        2nd-order reflected acoustic ray directions
        """
        reflects = []
        lines = []

        for wall1, wall2 in list(permutations(self.walls,2)):
            p_prime = mirror_point(p, wall1)
            o_prime = mirror_point(self.ego_vehicle, wall2)
            l = LineString([o_prime, p_prime])
            if l.intersects(wall1) and l.intersects(wall2):
                p_inter_1 = l.intersection(wall1)
                p_inter_2 = l.intersection(wall2)
                l1 = LineString([p, p_inter_1])
                l2 = LineString([p_inter_1, p_inter_2])
                l3 = LineString([p_inter_2, self.ego_vehicle])
                walls_1 = [w for w in self.walls if w != wall1]
                walls_2 = [w for w in self.walls if w != wall2 and w != wall1]
                if self.direct_line(l1, walls_1):
                    if self.direct_line(l2, walls_2):
                        loc = self.direct(p_inter_2, walls_2)
                        if len(loc) > 0:
                            reflects.append(loc)
                            lines += [l1,l2,l3]

        if not vis:
            return reflects
        else:
            return reflects, lines


    def diffract(self, p, vis=False):
        """
        Diffracted acoustic ray direction
        thres : maximum theta that is possible to diffract
        """
        diffracts = []
        lines = []
        if not self.direct_line(LineString([p, self.ego_vehicle])):
            for i,c in enumerate(self.corners):
                theta_corner = abs(math.atan(c.x/c.y))
                theta_p = abs(math.atan(p.x/p.y))
                if theta_p > theta_corner:
                    corner = Point(c.x, c.y+random.random())
                    v1 = [corner.x-p.x, corner.y-p.y]
                    v2 = [self.ego_vehicle.x-corner.x, self.ego_vehicle.y-corner.y]
                    theta = math.acos((v1[0]*v2[0] + v1[1]*v2[1])/(math.sqrt(v1[0]**2+v1[1]**2)*math.sqrt(v2[0]**2+v2[1]**2)))
                    
                    if theta < self.thres:
                        l1 = LineString([p, corner])
                        l2 = LineString([corner, self.ego_vehicle])
                        if self.direct_line(l1):
                            loc = self.direct(corner)
                            if len(loc) > 0:
                                diffracts.append(loc)
                                lines += [l1, l2]
        if not vis:
            return diffracts
        else:
            return diffracts, lines
    

class ParticleFilter:
    def __init__(self, n, _map, epsilon=0.05, weight_rate=[1,0.5,0.5,0.3]):
        self.n = n
        self.weight_rate = weight_rate
        self.particles = [Particle(self.weight_rate) for _ in range(self.n)]
        self.weight = np.array([1/self.n for _ in range(self.n)])
        
        self.time = 0
        self.epsilon = epsilon
        self.Map = _map
        # print(self.Map)

        self.prediction = None
        self.Max_prediction = None

        self.bbox_height = 3
        self.bbox_width = 5

        self.init_particles() 

    def init_particles(self):
        """
        Initialize particles.
        Use init instance when you want to set particle to specific state
        """
        for particle in self.particles:
            particle.random_initialize(self.Map)
        
    def propagate(self, t):
        """
        Motion propagation of particles
        t : time-step
        """
        for particle in self.particles:
            particle.move(t)

        self.time += t

    def update(self, z, weight_rate):
        """
        Motion update for each particles
        """
        for i in range(self.n):
            self.weight[i] = max(self.particles[i].measure_prob_addition(self.Map ,z, weight_rate), 0)

        if max(self.weight) == min(self.weight):
            self.weight = np.ones_like(self.weight)/self.weight.shape[0]
        
        self.predict()


    def resample(self):
        """
        Resampling
        Random particle with probability of epsilon
        """
        k = int((self.n)*(1-self.epsilon))
        sample_idx = random.choices(population=range(0,self.n), weights=self.weight, k=k)

        particles = []
        for idx in sample_idx:
            particles.append(deepcopy(self.particles[idx]))

        # random particle with probability of epsilon
        for _ in range(self.n-k):
            p = Particle(self.weight_rate)
            p.random_initialize(self.Map)
            particles.append(p)

        self.particles = particles


    def predict(self):
        """
        Predict sound source location
        """
        idx = np.argmax(self.weight)
        self.prediction = deepcopy(self.particles[idx])
        self.Max_prediction = deepcopy(self.particles[idx])

        # avg_with_intensity -> x, y 
        self.prediction.x = sum(x_val.x * w_val for x_val, w_val in zip(self.particles, self.weight)) / sum(self.weight)
        self.prediction.y = sum(y_val.y * w_val for y_val, w_val in zip(self.particles, self.weight)) / sum(self.weight)
        
        # prediction range
        self.prediction.x = max(-self.Map.width/2, min(self.Map.width/2, self.prediction.x))

    def distance(self,x1,x2,y1,y2) :
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def distance_rect(self, x1,x2, y1,y2,width,height) :
        if x1 <= x2 and x2 <= x1 + width :
            return 0
        else :
            return min(abs(x2-x1),abs(x2+width - x1))
        
    def average_particle(self,  x_points, y_points) :
        return round(sum(x_points) / len(x_points),2), round(sum(y_points)/len(y_points),2)
        

    def average_particle_with_intensity(self, x_points, y_points, w) :
        return round(sum(x_val * w_val for x_val, w_val in zip(x_points, w)) / sum(w),2), round(sum(y_val * w_val for y_val, w_val in zip(y_points, w)) / sum(w),2)


    def render(self, save_dir, is_ray = False):
        """
        Rendering
        """
        fig, ax = plt.subplots()
        
        # plot map
        x_map = [-self.Map.width/2, self.Map.width/2, self.Map.width/2, -self.Map.width/2, -self.Map.width/2]
        y_map = [0,0,self.Map.height, self.Map.height,0]
        ax.plot(x_map,y_map, color='black', linewidth=1.0)

        left_x_fill = [-self.Map.width / 2, self.Map.front_range[0], self.Map.front_range[0], -self.Map.width / 2]
        right_x_fill = [self.Map.front_range[1], self.Map.width / 2, self.Map.width / 2, self.Map.front_range[1]]
        y_fill = [0, 0, self.Map.front_height[0], self.Map.front_height[0]]

        ax.fill(left_x_fill, y_fill, 'gray', alpha=0.5)
        ax.fill(right_x_fill, y_fill, 'gray', alpha=0.5)
        if len(self.Map.walls) == 4:
            square_y = 6.5
        else : 
            x_upper_fill = [-self.Map.width / 2, self.Map.width / 2, self.Map.width / 2, -self.Map.width / 2]
            y_upper_fill = [self.Map.front_height[1], self.Map.front_height[1], self.Map.height, self.Map.height]
            ax.fill(x_upper_fill, y_upper_fill, 'gray', alpha=0.5)

        for wall in self.Map.walls: 
            x_wall, y_wall = wall.xy
            ax.plot(x_wall, y_wall, color='black', linewidth=1.0)

        # ego_vehicle
        ax.plot([0],[0], 'o',color='black', markersize=3)
        # ax.text(0, -0.2, 'ego vehicle', ha='center', va='top', fontsize=9)
        
        # particles
        x_p, y_p, dx_p, dy_p, weight_p = [], [], [], [], []

        for particle in self.particles:
            if -self.Map.width/2 < particle.x < self.Map.width/2 and self.Map.road_range[2] < particle.y < self.Map.height :
                x_p.append(particle.x)
                y_p.append(particle.y)
                dx_p.append(particle.v * math.cos(particle.head))
                dy_p.append(particle.v * math.sin(particle.head))
        

        ax.scatter(x_p, y_p, s=10, alpha=0.3, c='red')
        
        # prediction visualization -> Avg intensity
        ax.plot(self.prediction.x, self.prediction.y, c=(0,1,0,1), marker='o',markersize=7)

        # prediction visualization -> Max intensity
        # ax.plot(self.Max_prediction.x, self.Max_prediction.y, 'ro', markersize=7)

        if is_ray: # ploting ray
            p = Point(self.prediction.x, self.prediction.y)
            # p = Point(self.Max_prediction.x, self.Max_prediction.y)

            direct = []
        
            try:
                _,direct = self.Map.direct(p, vis=True)
            except:
                pass
            _,reflect_1st = self.Map.reflect_1st(p, vis=True)
            _,reflect_2nd = self.Map.reflect_2nd(p, vis=True)
            _,diffract = self.Map.diffract(p, vis=True)
           
            for i,l in enumerate(direct):
                x,y = l.xy
                ax.plot(x,y, c=(0,0,0,1), alpha=0.3, label="direct" if i==0 else "") # black
            for i,l in enumerate(reflect_1st):
                x,y = l.xy
                ax.plot(x,y, c=(0,1,1,1), alpha=0.3, label="1st reflect" if i==0 else "") # yellow
            for i,l in enumerate(reflect_2nd):
                x,y = l.xy
                ax.plot(x,y, c=(0,1,0,1), alpha=0.3, label="2nd reflect" if i==0 else "") # green
            if len(direct) == 0:
                for i,l in enumerate(diffract):
                    x,y = l.xy
                    ax.plot(x,y, c=(0,0,1,1), alpha=0.3, label="diffract" if i==0 else "") # blue

        plt.axis("equal")
        plt.title("%.2f sec"%self.time)
        # plt.xlim(-self.Map.width/2, self.Map.width/2)
        # plt.ylim(0, self.Map.height)
        # plt.xticks(range(int(-self.Map.width/2), int(self.Map.width/2)+1, 5))
        # plt.yticks(range(0,self.Map.height+1,5))
        plt.xticks([])
        plt.yticks([])

        plt.savefig(os.path.join(save_dir,'test_%06d.png'%(self.time*100)), bbox_inches = 'tight', pad_inches = 0)
        plt.close()


class ParticleFilter_FIX(ParticleFilter):
    """
    Particle with uniform spreaden fixed particles.
    """
    def __init__(self, n, _map, epsilon):
        super().__init__(n, _map, epsilon)
    
    def init_fixed_particles(self):
        """
        Initialize fixed particles.
        Use init instance when you want to set particle to specific state
        """
        self.w,self.h,self.h0 = self.Map.road_range
        self.n = (int(self.w)+1)*(int(self.h)+1)
        self.particles = [Particle() for _ in range(self.n)]
        self.weight = np.array([1/self.n for _ in range(self.n)])
        self.time = 0

        for i,p in zip(range(self.n),self.particles):
            wi = int(i / (int(self.h)+1))
            hi = int(i % (int(self.h)+1))
            p.x = wi * (self.w/int(self.w)) - self.w/2
            p.y = self.h0 + hi * (self.h/int(self.h))
            # self.head = random.uniform(0, 2*math.pi)
            p.head = random.choice([0, math.pi])
            p.v = random.random()*3 # 0~10m/s
            # self.w_head = (random.random()-0.5)
            p.w_head = 0
    
    def set_weight(self, weight):
        for p in self.particles:
            p.set_weight(weight)

    def propagate(self, t):
        self.time += t

    def trick(self):
        for i in range(self.w):
            self.weight[i*self.h:(i+1)*self.h] = np.average(self.weight[i*self.h:(i+1)*self.h])


    def predict(self):
        """
        Predict sound source location
        """

        # get particle mean
        idx = np.argmax(self.weight)
        particle = self.particles[idx]
        self.prediction = deepcopy(particle)
        theta = math.atan(particle.x/particle.y)


        # theta = math.atan2(self.particles[idx].x, self.particles[idx].y)
        theta_left = math.atan(self.Map.corners[0].x/self.Map.corners[0].y)
        theta_right = math.atan(self.Map.corners[1].x/self.Map.corners[1].y)

        # print(np.rad2deg(theta), np.rad2deg(theta_left), np.rad2deg(theta_right))
        if theta < theta_left:
            return 'left'
        elif theta > theta_right:
            return 'right'
        else:
            return 'front'

    def render(self, save_dir, name):
        """
        Rendering
        """
        fig, ax = plt.subplots()
        
        # plot map
        x_map = [-self.Map.width/2, self.Map.width/2, self.Map.width/2, -self.Map.width/2, -self.Map.width/2]
        # x_map = [-self.Map.width_left, self.Map.width_right, self.Map.width_right, -self.Map.width_left, -self.Map.width_left]
        y_map = [0,0,self.Map.height, self.Map.height,0]
        ax.plot(x_map,y_map, color='black', linewidth=3.0)

        for wall in self.Map.walls:
            x_wall, y_wall = wall.xy
            ax.plot(x_wall, y_wall, color='black', linewidth=2.0)

        # ego_vehicle
        ax.plot([0],[0], 'bo', markersize=10)

        # particles
        x_p, y_p, dx_p, dy_p = [], [], [], []
        for particle, w in zip(self.particles, self.weight):
            color = [[1,0.9*(1-w/self.weight.max()),0.9*(1-w/self.weight.max()),1]]
            s = 40 * w / self.weight.max()
            ax.scatter([particle.x], [particle.y], s=s, c=color)

        plt.axis("equal")
        plt.axis('off')
        plt.title(name)
        # plt.xlim(-self.Map.width_left, self.Map.width_right)
        # plt.ylim(0, self.Map.height)
        # plt.xticks(range(int(-self.Map.width_left), int(self.Map.width_right)+1, 5))
        # plt.yticks(range(0,self.Map.height+1,5))
        
        # plt.savefig(os.path.join(save_dir,'test_{}.png'.format(name)))
        # plt.close()
        plt.xlim(-self.Map.width/2, self.Map.width/2)
        plt.ylim(0, self.Map.height)
        plt.xticks(range(int(-self.Map.width/2), int(self.Map.width/2)+1, 5))
        plt.yticks(range(0,self.Map.height+1,5))
        
        plt.savefig(os.path.join(save_dir,'test_{}.png'.format(name)))
        plt.close()
