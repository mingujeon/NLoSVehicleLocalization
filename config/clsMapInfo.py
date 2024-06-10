from shapely.geometry import LineString, Point
import math

#####################################
## Classification dataset MapInfo ###
#####################################
corner_offset = 0.1

# T junction with a wall

SA_cor_left = -2.8
SA_cor_right = 2.8
SA_front_left = 4.45
SA_front_right = 4.45
SA_wall = SA_front_left+5.45
SA_width = 12.5

SA_map_info = {
    "width" : 2*SA_width,
    "height" : 13,
    "walls" : [LineString([(SA_cor_left,0),(SA_cor_left,SA_front_left)]), 
            LineString([(SA_cor_right,0),(SA_cor_right,SA_front_right)]),
            LineString([(-SA_width,SA_wall),(SA_width,SA_wall)]),
            LineString([(SA_cor_left,SA_front_left),(-SA_width,SA_front_left)]),
            LineString([(SA_cor_right,SA_front_right),(SA_width,SA_front_right)])],
    "corners" : [Point(SA_cor_left+corner_offset,SA_front_left+corner_offset), Point(SA_cor_right-corner_offset,SA_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SA_cor_left, SA_cor_right],
    "road_range" : [2*SA_width, SA_wall-SA_front_left-0.2, SA_front_left+0.1],
    "front_height" : [SA_front_left, SA_wall]
}

# T junction without a wall

SB_cor_left = -2.78
SB_cor_right = 2.78
SB_front_left = 4.5
SB_front_right = 4.5
SB_width = 10

SB_map_info = {
    "width" : 2*SB_width,
    "height" : 13,
    "walls" : [LineString([(SB_cor_left,0),(SB_cor_left,SB_front_left)]), 
            LineString([(SB_cor_right,0),(SB_cor_right,SB_front_right)]),
            LineString([(SB_cor_left,SB_front_left),(-SB_width,SB_front_left)]),
            LineString([(SB_cor_right,SB_front_right),(SB_width,SB_front_right)])],
    "corners" : [Point(SB_cor_left+corner_offset,SB_front_left+corner_offset), Point(SB_cor_right-corner_offset,SB_front_right+corner_offset)],
    "diffraction_angle_threshold" : math.pi / 3,

    "front_range" : [SB_cor_left, SB_cor_right],
    "road_range" : [2*SB_width, 20-SB_front_left-0.2, SB_front_left+0.1],
    "front_height" : [SB_front_left]
}
