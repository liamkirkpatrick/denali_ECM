#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:31:42 2024

@author: Liam
"""

#%% Import packages

import pandas as pd
import numpy as np
import math

#%% User inputs

path_to_data = '../../data/processed_data/'
metadata = 'metadata.csv'

#%% Define ECM Class

class ECM:
    
    def __init__(self,core,section,face,ACorDC):
        
        # open metadata csv
        meta = pd.read_csv(path_to_data+metadata)
        row = meta.loc[meta['core']==core]
        row = row.loc[row['section']==section]
        row = row.loc[row['face']==face]
        row = row.loc[row['ACorDC']==ACorDC]
        
        
        # assign core components
        self.time = row['time'].values[0]
        self.y_left = row['Y_left'].values[0]
        self.y_right = row['Y_right'].values[0]
        self.core = row['core'].values[0]
        self.section = row['section'].values[0]
        self.face = row['face'].values[0]
        self.ACorDC = row['ACorDC'].values[0]
        
        # open core csv
        fname = self.core+'-'+str(self.section)+'-'+self.face+'-'+self.ACorDC+'.csv'
        raw = pd.read_csv(path_to_data+fname)
        

        
        # assign vectors
        self.meas = raw['meas'].to_numpy()
        self.y = raw['Y_dimension(mm)'].to_numpy()
        self.button = raw['Button'].to_numpy()
        self.depth = raw['True_depth(m)'].to_numpy()
        self.y_vec = np.unique(self.y)
        if 'button_raw' in raw.columns:
            self.button_raw = raw['button_raw'].to_numpy()
        else:
            self.button_raw = None
            
        # remove tracks that are incomplete
        lenth = []
        approx_length = max(self.depth) - min(self.depth)
        for y in self.y_vec:
            idx = self.y==y
            track_length = max(self.depth[idx]) - min(self.depth[idx])
            # remove tracks not within 1cm of overall length
            if abs(approx_length - track_length) > 0.01:
                self.meas = self.meas[np.invert(idx)]
                self.y = self.y[np.invert(idx)]
                self.button = self.button[np.invert(idx)]
                self.depth = self.depth[np.invert(idx)]
                self.y_vec = self.y_vec[self.y_vec!=y]
                if 'button_raw' in raw.columns:
                    self.button_raw = self.button_raw[np.invert(idx)]
        
        # assign status
        self.issmoothed = False
        
    def smooth(self,window):
        # takes as input smoothing window (in mm)
        
        # convert window to m
        window=window/1000
        
        # get spacing between points
        vec = self.depth[self.y==self.y_vec[0]]
        dist = []
        for i in range(len(vec)-1):
            dist.append(vec[i+1]-vec[i])
        
        # make vector of depths to interpolate onto
        dvecmin = min(self.depth) + 2*window/3
        dvecmax = max(self.depth) - 2*window/3
        dvec_num = math.floor((dvecmax-dvecmin) / abs(np.mean(dist)))
        depth_vec = np.linspace(dvecmin,dvecmax,dvec_num)
        #self.dvec = np.flip(depth_vec)
        
        # make empty smooth vectors
        depth_smooth = []
        meas_smooth = []
        button_smooth = []
        y_smooth = []
        
        # loop through all tracks
        for y in self.y_vec:
            
            # index within track
            idx = self.y == y
            
            dtrack = self.depth[idx]
            mtrack = self.meas[idx]
            btrack = self.button[idx]
            
            # loop through all depths
            for d in depth_vec:
                
                # find index of all points within window of this depth
                didx = (dtrack >= d-window/2) * (dtrack <= d+window/2)
                
                # save values
                depth_smooth.append(d)
                meas_smooth.append(np.median(mtrack[didx]))
                y_smooth.append(y)
                if sum(btrack[didx])>0:
                    button_smooth.append(1)
                else:
                    button_smooth.append(0)
                
        # save smooth values
        self.depth_s = np.flip(np.array(depth_smooth))
        self.meas_s = np.flip(np.array(meas_smooth))
        self.button_s = np.flip(np.array(button_smooth))
        self.y_s = np.flip(np.array(y_smooth))
        
        self.issmoothed = True
        
    def rem_ends(self,clip):
        
        # convert clip to m
        clip = clip/1000
        
        # find index within clip
        dmin = min(self.depth)
        dmax = max(self.depth)
        idx = (self.depth>=dmin+clip) * (self.depth<=dmax-clip)
        
        self.meas = self.meas[idx]
        self.y = self.y[idx]
        self.button = self.button[idx]
        self.depth = self.depth[idx]
        
        
        # check if smooth exists
        if hasattr(self,'y_s'):
            
            # find index within clip
            dmin = min(self.depth_s)
            dmax = max(self.depth_s)
            idx = (self.depth_s>=dmin+clip) * (self.depth_s<=dmax-clip)
            
            self.meas_s = self.meas_s[idx]
            self.y_s = self.y_s[idx]
            self.button_s = self.button_s[idx]
            self.depth_s = self.depth_s[idx]
            
    
    # normalize outside magnitude to match inner tracks
    def norm_outside(self):
                
        # calculate average accross main track
        norm_idx1 = self.y>self.y_vec[0]
        norm_idx2 = self.y<self.y_vec[-1]
        norm_idx = norm_idx1*norm_idx2
        norm = np.mean(self.meas[norm_idx])
        
        # now loop through each outside track
        for ytrack in [self.y_vec[0],self.y_vec[-1]]:
            track_idx = self.y == ytrack
            track_ave = np.mean(self.meas[track_idx])
            self.meas[track_idx] = self.meas[track_idx] * norm / track_ave
            if self.issmoothed:
                strack_idx = self.y_s == ytrack
                self.meas_s[strack_idx] = self.meas_s[strack_idx] * norm / track_ave
                
            print(norm/track_ave)
            
    # normalize all tracks
    def norm_all(self):
        
        # loop through all tracks
        for ytrack in self.y_vec:
            
            button_idx = self.button == 0
            track_idx = self.y == ytrack
            trackave = np.mean(self.meas[track_idx*button_idx])
            self.meas[track_idx] = self.meas[track_idx] / trackave
            
            # normalize smooth track if it exists
            if self.issmoothed:
                
                sbutton_idx = self.button_s == 0
                strack_idx = self.y_s == ytrack
                strackave = np.mean(self.meas_s[strack_idx*sbutton_idx])
                self.meas_s[strack_idx] = self.meas_s[strack_idx] / strackave

        
        
#%% Test

if __name__ == "__main__":
    
    #test = ECM('alhic2201','10_1','t','AC')
    test = ECM('alhic2302','159_3','t','AC')
    
    test.smooth(1)
    
    test.rem_ends(1)
    
    test.smooth(1)
    
    #test.norm_outside()
    
    test.norm_all()
    
