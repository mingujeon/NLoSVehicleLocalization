from shapely.geometry import LineString, Point
import math

#####################################
## Tracking dataset MapInfo #########
#####################################
corner_offset = 0.05

IA_cor_left = -0.85
IA_cor_right = 0.85
IA_front_left = 2.5
IA_front_right = 2.5
IA_wall = IA_front_left+6.85
IA_width = 10.0
IA_map_info = {
    "width" : 2*IA_width,
    "height" : 10,
    "walls" : [LineString([(IA_cor_left,0),(IA_cor_left,IA_front_left)]), 
            LineString([(IA_cor_right,0),(IA_cor_right,IA_front_right)]),
            LineString([(IA_cor_left,IA_front_left),(-IA_width,IA_front_left)]),
            LineString([(IA_cor_right,IA_front_right),(IA_width,IA_front_right)]),
            LineString([(-IA_width,IA_wall),(IA_width,IA_wall)])],
    "corners" : [Point(IA_cor_left+corner_offset,IA_front_left+corner_offset), Point(IA_cor_right-corner_offset,IA_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "road_range" : [2*IA_width, IA_wall-IA_front_left-3.8, IA_front_left+0.1]
}

IB_cor_left = -0.95
IB_cor_right = 0.80
IB_front_left = 1.95
IB_front_right = 1.95
IB_wall = IB_front_left+2.0
IB_width = 6.0
IB_map_info = {
    "width" : 2*IB_width,
    "height" : 5,
    "walls" : [LineString([(IB_cor_left,0),(IB_cor_left,IB_front_left)]), 
            LineString([(IB_cor_right,0),(IB_cor_right,IB_front_right)]),
            LineString([(IB_cor_left,IB_front_left),(-IB_width,IB_front_left)]),
            LineString([(IB_cor_right,IB_front_right),(IB_width,IB_front_right)]),
            LineString([(-IB_width,IB_wall),(IB_width,IB_wall)])],
    "corners" : [Point(IB_cor_left+corner_offset,IB_front_left+corner_offset), Point(IB_cor_right-corner_offset,IB_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "road_range" : [2*IB_width, IB_wall-IB_front_left-0.1, IB_front_left+0.1]
}

IC_cor_left = -1.0
IC_cor_right = 0.90
IC_front_left = 2.15
IC_front_right = 10.0
IC_wall = IC_front_left+1.75
IC_width = 6.0
IC_map_info = {
    "width" : 2*IC_width,
    "height" : 12,
    "walls" : [LineString([(IC_cor_left,0),(IC_cor_left,IC_front_left)]), 
            LineString([(IC_cor_right,0),(IC_cor_right,IC_front_right)]),
            LineString([(IC_cor_left,IC_front_left),(-IC_width,IC_front_left)]),
            LineString([(IC_cor_left,IC_wall),(IC_cor_left,IC_front_right)]),
            LineString([(-IC_width,IC_wall),(IC_cor_left,IC_wall)])],
    "corners" : [Point(IC_cor_left+corner_offset,IC_front_left+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "road_range" : [2*IC_width, IC_front_right-IC_front_left-0.1, IC_front_left+0.1]
}

OA_cor_left = -3.5
OA_cor_right = 2.0
OA_front_left = 3.8
OA_front_right = 4.9
OA_wall = OA_front_left+10.75
OA_wall_left = -15.0
OA_wall_right = 15.0
OA_width = 12.5
OA_map_info = {
    "width" : 2*OA_width,
    "height" : 15,
    "walls" : [LineString([(OA_cor_left,0),(OA_cor_left,OA_front_left)]), 
            LineString([(OA_cor_right,0),(OA_cor_right,OA_front_right)]),
            LineString([(OA_cor_left,OA_front_left),(-OA_width,OA_front_left)]),
            LineString([(OA_cor_right,OA_front_right),(OA_width,OA_front_right)]),
            LineString([(OA_wall_left,OA_wall),(OA_wall_right,OA_wall)])],
    "corners" : [Point(OA_cor_left+corner_offset,OA_front_left+corner_offset), Point(OA_cor_right-corner_offset,OA_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "road_range" : [2*OA_width, OA_wall-OA_front_left-0.1, OA_front_left+0.1]
}