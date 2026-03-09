import os
import sys
import tkinter as tk
from tkinter import filedialog as fd
import numpy as np
import pandas as pd
import sympy.geometry as spg
import matplotlib.path as mplPath
from datetime import datetime


def footprints(cam,sensor,base_elev): 
    """
    This function calculates the instantaneous field of view (IFOV) for 
    the camera(s) that are passed.\n
    Vars:\n
    \t cam = pandas dataframe (n x ~6, fields: x,y,z,yaw,pitch,roll)\n
    \t sensor = pandas dataframe (1 x 3, fields: focal, sensor_x, sensor_y):
    \t focal length (mm), sensor x dim (mm), sensor y dim (mm)\n
    \t base_elev = average elevation of your site (meters, or in the same
    \t measure as your coordinates)\n
    Creates approx. coordinates for sensor
    corners (north-oriented and zero pitch) at the camera's x,y,z. Rotates
    the sensor coords in 3D space to the camera's pitch and yaw angles (roll
    angles are ignored for now) and projects corner rays through the camera 
    x,y,z to a approx ground plane. The intersection of the rays with the
    ground are the corners of the photo footprint.\n
    *** Photos that have picth angles that cause the horizon to be visable will
    cause the UL and UR path coordniates to wrong. These cameras are 
    disreguarded and the footprint will be set to NaN in the output.***\n 
    RETURNS: footprints = Pandas dataframe (n x 1) of Matplotlib Path objects()
    """
    # Setup DF to house camera footprint polygons
    footprints = pd.DataFrame(np.zeros((cam.shape[0],1)), columns=['fov'])
    
    # convert sensor dimensions to meters, divide x/y for corner coord calc
    f = sensor.focal[0] * 0.001
    sx = sensor.sensor_x[0] / 2 * 0.001
    sy = sensor.sensor_y[0] / 2 * 0.001

    # calculate the critical pitch (in degrees) where the horizon will be 
    #   visible with the horizon viable, the ray projections go backward 
    #   and produce erroneous IFOV polygons (90 - 0.5*vert_fov)
    crit_pitch = 90 - np.rad2deg(np.arctan(sy / f))
    
    # User Feedback
    print("Proccesing Camera IFOVs (%i total)..." %(cam.shape[0]))
    sys.stdout.flush()
     
    # for each camera...
    for idx, row in cam.iterrows():
        
        # check is the camera pitch is over the critical value
        if row.pitch < crit_pitch:
            
            # sensor corners (UR,LR,LL,UL), north-oriented and zero pitch
            corners = np.array([[row.x+sx,row.y-f,row.z+sy],
                               [row.x+sx,row.y-f,row.z-sy],
                               [row.x-sx,row.y-f,row.z-sy],
                               [row.x-sx,row.y-f,row.z+sy]])
            
            # offset corner points by cam x,y,z for rotation
            cam_pt = np.atleast_2d(np.array([row.x, row.y, row.z]))
            corner_p = corners - cam_pt
    
            # get pitch and yaw from the camera, convert to radians
            pitch = np.deg2rad(90.0-row.pitch)
            roll = np.deg2rad(row.roll)
            yaw = np.deg2rad(row.yaw)
            
            # setup picth rotation matrix (r_x) and yaw rotation matrix (r_z)
            r_x = np.matrix([[1.0,0.0,0.0],
                             [0.0,np.cos(pitch),-1*np.sin(pitch)],
                             [0.0,np.sin(pitch),np.cos(pitch)]])
                             
            r_y = np.matrix([[np.cos(roll),0.0,np.sin(roll)],
                             [0.0,1.0,0.0],
                             [-1*np.sin(roll),0.0,np.cos(roll)]])
            
            r_z =  np.matrix([[np.cos(yaw),-1*np.sin(yaw),0],
                              [np.sin(yaw),np.cos(yaw),0],
                              [0,0,1]])
            
            # rotate corner_p by r_x, then r_z, add back cam x,y,z offsets
            # produces corner coords rotated for pitch and yaw
            p_pr = np.matmul(np.matmul(corner_p, r_x),r_y)            
            p_out = np.matmul(p_pr, r_z) + cam_pt
            
            # GEOMETRY
            # Set Sympy 3D point for the camera and a 3D plane for intersection
            cam_sp = spg.Point3D(row.x, row.y, row.z)
            plane = spg.Plane(spg.Point3D(row.x, row.y, base_elev),
                                      normal_vector=(0,0,1))
            
            # blank array for footprint intersection coords
            inter_points = np.zeros((corners.shape[0],2))
            
            # for each sensor corner point
            idx_b = 0
            for pt in np.asarray(p_out):
                
                # create a Sympy 3D point and create a Sympy 3D ray from 
                #   corner point through camera point
                pt_sp = spg.Point3D(pt[0],pt[1],pt[2])
                ray = spg.Ray3D(pt_sp,cam_sp)
                
                # calculate the intersection of the ray with the plane                
                inter_pt = plane.intersection(ray)
                
                # Extract out the X,Y coords fot eh intersection point
                #   ground intersect points will be in this order (LL,UL,UR,LR)
                inter_points[idx_b,0] = inter_pt[0].x.evalf()
                inter_points[idx_b,1] = inter_pt[0].y.evalf()
                
                idx_b += 1
        
        # if crit_pitch is exceeded set inter_points to NaN
        else:
            inter_points = np.full((4,2),np.nan)
        
        # append inter_points to footprints as a matplotlib path object
        footprints.fov[idx] = mplPath.Path(inter_points)
        
        # User feedback
        if (idx+1) % 10 == 0:
            print("%i cameras processed..." %(idx+1))
            sys.stdout.flush()
    
    return footprints
# END - footprints