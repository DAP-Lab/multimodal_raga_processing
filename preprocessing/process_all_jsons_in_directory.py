import pandas as pd
import numpy as np
import json
from sys import argv
import os
import scipy.signal as signal
import datetime
import re


'''
this code will do the following for all jsons in a directory:-
1. combine the individual json files into a time series
2. do appropiate interpolations for missing values
3. do interpolations for resampling at 10 second intervals
4. do low pass filtering
5. do normalization
5. rename the file appropiately
'''
# Source:- https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
#  Result for BODY_25 (25 body parts consisting of COCO + foot)
#  const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
#      {0,  "Nose"},
#      {1,  "Neck"},
#      {2,  "RShoulder"},
#      {3,  "RElbow"},
#      {4,  "RWrist"},
#      {5,  "LShoulder"},
#      {6,  "LElbow"},
#      {7,  "LWrist"},
#      {8,  "MidHip"},
#      {9,  "RHip"},
#      {10, "RKnee"},
#      {11, "RAnkle"},
#      {12, "LHip"},
#      {13, "LKnee"},
#      {14, "LAnkle"},
#      {15, "REye"},
#      {16, "LEye"},
#      {17, "REar"},
#      {18, "LEar"},
#      {19, "LBigToe"},
#      {20, "LSmallToe"},
#      {21, "LHeel"},
#      {22, "RBigToe"},
#      {23, "RSmallToe"},
#      {24, "RHeel"},
#      {25, "Background"}
#  };
class ProcessKeyPoints(object):
    def __init__(self):
        self.setup()

    def run(self):
        list_of_directories=[os.path.join(self.json_files_dir,x) for x in os.listdir(self.json_files_dir) if not re.search('zip',x)]
        #print (list_of_directories)
        for each_directory in list_of_directories:
            print ("Started processing ",each_directory," at ",datetime.datetime.now())
            try:
                self.make_time_series(each_directory,save=True)
            except Exception as e:
                print ("Failed for ",each_directory," with ",str(e))
                pass


    def setup(self):
        self.body_keypoints_master={
            0:  "Nose",
            1:  "Neck",
            2:  "RShoulder",
            3:  "RElbow",
            4:  "RWrist",
            5:  "LShoulder",
            6:  "LElbow",
            7:  "LWrist",
            8:  "MidHip",
            9:  "RHip",
            10: "RKnee",
            11: "RAnkle",
            12: "LHip",
            13: "LKnee",
            14: "LAnkle",
            15: "REye",
            16: "LEye",
            17: "REar",
            18: "LEar",
            19: "LBigToe",
            20: "LSmallToe",
            21: "LHeel",
            22: "RBigToe",
            23: "RSmallToe",
            24: "RHeel",
            25: "Background"
        }
        #  
        self.fps=24.0
        self.time_per_frame=np.round(1/self.fps,3)

        # directory_to_process=argv[1]

        self.output_dir_pose='./pose_keypoints_dir'
        self.output_dir_all='./keypoints_all'
        self.output_dir_all_non_normalized='./keypoints_non_normalized'
        self.json_files_dir='./outputJsonTar'

        self.image_size_x=1920.0
        self.image_size_y=1080.0

        os.makedirs(self.output_dir_pose,exist_ok=True)
        os.makedirs(self.output_dir_all,exist_ok=True)
        os.makedirs(self.output_dir_all_non_normalized,exist_ok=True)


    def convert_openpose_list_to_kv(self,input_list,frame_number,type_of_file):
        return_dict={}
        for i in range(len(input_list)):
            index=i//3
            if i%3==0:
                return_dict[index]={}
                return_dict[index]['x']=input_list[i]
            elif i%3==1:   
                return_dict[index]['y']=input_list[i]
            else:
                return_dict[index]['conf']=input_list[i]
        if type_of_file=='body':
            return_dict={self.body_keypoints_master[k]:v for k,v in return_dict.items()}
        # else:
        #     return_dict={type_of_file+'_'+str(k):v for k,v in return_dict.items()}
        return_dict['frame_number']=frame_number
        return_dict['time']=frame_number*self.time_per_frame
        return return_dict

    def create_empty_openpose_data(self,type_of_file,frame_number):
        return_dict={}
        if type_of_file=='body':
            num_points=25
        else:
            num_points=21
        for i in range(num_points):
            return_dict[i]={}
            return_dict[i]['x']=np.nan
            return_dict[i]['y']=np.nan
            return_dict[i]['conf']=0.0
        if type_of_file=='body':
            return_dict={self.body_keypoints_master[k]:v for k,v in return_dict.items()}
        # else:
        #     return_dict={type_of_file+'_'+str(k):v for k,v in return_dict.items()}
        return_dict['frame_number']=frame_number
        return_dict['time']=frame_number*self.time_per_frame
        return return_dict    

    def do_interpolate_and_lpf(self,list_of_kepoints,type_of_file):
        '''
        Interpolate for missing values and use low pass filtering for 
        '''
        output_list=[]
        names=[]
        keypoints=list_of_kepoints[0].keys()
        #print (keypoints)
        #print (list_of_kepoints[0:5])
        timeseries_input=[x['time'] for x in list_of_kepoints]
        for each_keypoint in keypoints:
            if each_keypoint in ['frame_number','time']:
                output_list.append(pd.Series([p[each_keypoint] for p in list_of_kepoints]))
                names.append(each_keypoint)
                continue
            if each_keypoint not in ['Neck','MidHip']:
                x_list=np.array([np.nan if p[each_keypoint]['conf']<1e-6 else p[each_keypoint]['x'] for p in list_of_kepoints])
                x_list[x_list<0]=np.nan
                x_list[x_list>self.image_size_x]=np.nan
                y_list=np.array([np.nan if p[each_keypoint]['conf']<1e-6 else p[each_keypoint]['y'] for p in list_of_kepoints])
                y_list[y_list<0]=np.nan
                y_list[y_list>self.image_size_y]=np.nan
            else:
                x_list=np.array([p[each_keypoint]['x'] for p in list_of_kepoints])
                x_list[x_list<0]=np.nan
                x_list[x_list>self.image_size_x]=np.nan
                y_list=np.array([p[each_keypoint]['y'] for p in list_of_kepoints])
                y_list[y_list<0]=np.nan
                y_list[y_list>self.image_size_y]=np.nan
            x_list_pd=pd.Series(x_list)
            x_list_pd=x_list_pd.interpolate(method='linear')
            x_list_pd=pd.Series(signal.savgol_filter(x_list_pd.values,13,4))
            if type_of_file != 'body':
                col_name_x=type_of_file+'_'+str(each_keypoint)+'_x'
            else:
                col_name_x=str(each_keypoint)+'_x'
            y_list_pd=pd.Series(y_list)
            y_list_pd=pd.Series(signal.savgol_filter(y_list_pd.values,13,4))
            y_list_pd=y_list_pd.interpolate(method='linear',order=3)
            if type_of_file != 'body':
                col_name_y=type_of_file+'_'+str(each_keypoint)+'_y'
            else:
                col_name_y=str(each_keypoint)+'_y'
            output_list.append(x_list_pd)
            names.append(col_name_x)
            output_list.append(y_list_pd)
            names.append(col_name_y)
        # print ([type(k) for k in output_list])
        output_df=pd.concat(output_list,axis=1)
        output_df.columns=names
        return output_df

    def normalize_data(self,all_keypoints_pd):
        all_keypoints_pd_normalized=all_keypoints_pd.copy(deep=True)
        cols_x=[col for col in list(all_keypoints_pd.columns) if re.search('_x',col)]
        x_vals=all_keypoints_pd[cols_x].values
        x_vals[x_vals>self.image_size_x]=np.nan
        x_vals[x_vals<0]=np.nan
        cols_y=[col for col in list(all_keypoints_pd.columns) if re.search('_y',col)]
        y_vals=all_keypoints_pd[cols_y].values
        y_vals[y_vals>self.image_size_y]=np.nan
        y_vals[y_vals<0]=np.nan
        xmax=np.nanmax(x_vals)
        xmin=np.nanmin(x_vals)
        ymax=np.nanmax(y_vals)
        ymin=np.nanmin(y_vals)
        midHip_x=all_keypoints_pd['MidHip_x'].values
        midHip_x[midHip_x>self.image_size_x]=np.nan
        midHip_x[midHip_x<0]=np.nan
        neck_x=all_keypoints_pd['Neck_x'].values
        neck_x[neck_x>self.image_size_x]=np.nan
        neck_x[neck_x<0]=np.nan
        midHip_x_mean=np.nanmean(midHip_x)
        neck_x_mean=np.nanmean(neck_x)
        xc = (np.nanmean(neck_x_mean) + np.nanmean(midHip_x_mean)) / 2
        width = 2 * max(xc - xmin, xmax - xc)
        height = ymax - ymin
        for each_col in list(all_keypoints_pd.columns):
            if each_col in ['frame_number','time']:
                continue
            else:
                if re.search('_x',each_col):
                    all_keypoints_pd_normalized[each_col]= (all_keypoints_pd[each_col]-xmin)/width
                elif re.search('_y',each_col):
                    all_keypoints_pd_normalized[each_col]= (all_keypoints_pd[each_col]-ymin)/height
        return all_keypoints_pd_normalized

    def resample_dataframe(self,input_df):
        output_series_list=[]
        output_names_list=[]
        input_df.drop('frame_number',axis=1,inplace=True)
        max_time=np.max(input_df['time'])
        num_samples=int(max_time*100)
        resampled_time=np.linspace(0, num_samples, num_samples, endpoint=False)
        resampled_time=resampled_time/100
        resampled_time_series=pd.Series(resampled_time)
        output_series_list.append(resampled_time_series)
        output_names_list.append('time')
        for each_column in input_df.columns:
            if each_column=='time':
                continue
            this_series=input_df[each_column]
            if this_series.isnull().all():
                continue
            resampled=signal.resample(this_series,num_samples) 
            resampled_series=pd.Series(resampled) 
            output_series_list.append(resampled_series)
            output_names_list.append(each_column) 
        output_df=pd.concat(output_series_list,axis=1)
        output_df.columns=output_names_list
        return output_df


    def make_time_series(self,directory_to_process_full_name,save=False):
        directory_to_process=os.path.basename(directory_to_process_full_name)
        directory_to_process=re.sub(' ','_',directory_to_process)
        directory_to_process=re.sub('\(','_',directory_to_process)
        directory_to_process=re.sub('\)','_',directory_to_process)
        list_of_files=sorted([os.path.join(directory_to_process_full_name,x) for x in os.listdir(directory_to_process_full_name)])
        pose_keypoints_list=[]
        hand_left_keypoints_list=[]
        hand_right_keypoints_list=[]
        frame_number=0
        for each_file in list_of_files:
            # print (each_file) 
            with open(each_file,'r') as f:
                openpose_data=json.load(f)   
            if len(openpose_data['people'])==0:
                pose_keypoints_2d=self.create_empty_openpose_data('body',frame_number)
                hand_left_keypoints_2d=self.create_empty_openpose_data('left_hand',frame_number)
                hand_right_keypoints_2d=self.create_empty_openpose_data('right_hand',frame_number)
                pose_keypoints_list.append(pose_keypoints_2d)
                hand_left_keypoints_list.append(hand_left_keypoints_2d)
                hand_right_keypoints_list.append(hand_right_keypoints_2d)
                frame_number=frame_number+1
                continue
            pose_keypoints_2d=self.convert_openpose_list_to_kv(openpose_data['people'][0]['pose_keypoints_2d'],frame_number,'body')
            #pose_keypoints_2d={body_keypoints_master[k]:v for k,v in pose_keypoints_2d.items()}
            hand_left_keypoints_2d=self.convert_openpose_list_to_kv(openpose_data['people'][0]['hand_left_keypoints_2d'],frame_number,'right_hand')
            hand_right_keypoints_2d=self.convert_openpose_list_to_kv(openpose_data['people'][0]['hand_right_keypoints_2d'],frame_number,'left_hand')
            pose_keypoints_list.append(pose_keypoints_2d)
            hand_left_keypoints_list.append(hand_left_keypoints_2d)
            hand_right_keypoints_list.append(hand_right_keypoints_2d)
            frame_number=frame_number+1
        pose_keypoints_pd=self.do_interpolate_and_lpf(pose_keypoints_list,'body') #pd.DataFrame(pose_keypoints_list)
        hand_left_keypoints_pd=self.do_interpolate_and_lpf(hand_left_keypoints_list,'Left_hand') #pd.DataFrame(hand_left_keypoints_list)
        hand_right_keypoints_pd=self.do_interpolate_and_lpf(hand_right_keypoints_list,'Right_hand') #pd.DataFrame(hand_right_keypoints_list)
        if save:
            output_file_pose=os.path.join(self.output_dir_pose,directory_to_process.strip()+'_pose.csv')
            output_file_all_non_normalized=os.path.join(self.output_dir_all_non_normalized,directory_to_process.strip()+'_all_nonnormalized.csv')
            output_file_all=os.path.join(self.output_dir_all,directory_to_process.strip()+'_all.csv')
            hand_left_keypoints_pd.drop(['frame_number','time'], axis=1,inplace=True)
            hand_right_keypoints_pd.drop(['frame_number','time'], axis=1,inplace=True)
            output_pd_all=pd.concat([pose_keypoints_pd,hand_left_keypoints_pd,hand_right_keypoints_pd],axis=1)    
            output_pd_all.to_csv(output_file_all_non_normalized,index=False,float_format='%.3f')
            output_pd_resampled=self.resample_dataframe(output_pd_all)
            output_pd_normalized=self.normalize_data(output_pd_resampled)
            pose_keypoints_pd_keep=output_pd_normalized[['RElbow_x','RElbow_y','RWrist_x','RWrist_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y','time']]
            pose_keypoints_pd_keep.to_csv(output_file_pose,index=False,float_format='%.3f')        
            output_pd_normalized.to_csv(output_file_all,index=False,float_format='%.3f')

processOpenPose=ProcessKeyPoints()
processOpenPose.run()