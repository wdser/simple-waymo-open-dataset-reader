# adapted from 'visualize_pcl.py'

import os
import sys

import numpy as np
import pandas as pd

import cv2
import pdb
import json
import pandas as pd
import tensorflow as tf
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict

from tqdm import tqdm
from pathlib import Path
from pyntcloud import PyntCloud

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils

# if len(sys.argv) != 3:
#     print("""Usage: python extract_merged_pointclouds.py <tf_records_path> <save_path>
# Extracting the merged point clouds from all sensors from all .tfrecord files within the given <tf_records_path>
# and saving each frame as an individual .ply file under the given <save_path>.
# The whole extraction process can take up to 3 days (~70 hours).""")
#     sys.exit(0)

path = filename = sys.argv[1]
save_path_root = filename = sys.argv[2]

total_num_frames = 0

files = sorted(os.listdir(path))
files = [file for file in files if file.endswith(".tfrecord")]


for file in tqdm(files):

    filename = os.path.join(path, file)

    datafile = WaymoDataFileReader(filename)
    table = datafile.get_record_table()

    frames = iter(datafile)

    frame_number = 0

    for frame in frames:
        timestamp = frame.timestamp_micros
        context = frame.context
        pose = frame.pose
        laser_labels = frame.laser_labels
        projected_lidar_labels = frame.projected_lidar_labels
        camera_labels = frame.camera_labels

        entry_dir = save_path_root + str(filename.split('/')[-1]).replace('.tfrecord', '')+ '/' + str(timestamp) + '/'
        # print(entry_dir)
        Path(entry_dir).mkdir(exist_ok=True)
        # continue

        # load and save pose
        pose_json_str = json_format.MessageToJson(pose)
        # context_json_bytes = context_json_str.encode('utf-8')
        with open(entry_dir+"pose.json", "w") as file:
            file.write(pose_json_str)
 
        # load and save context
        context_json_str = json_format.MessageToJson(context)
        # context_json_bytes = context_json_str.encode('utf-8')
        with open(entry_dir+"context.json", "w") as file:
            file.write(context_json_str)

        # load and save laser_labels
        laser_label_msgs = [item for item in laser_labels]
        serializable_laser_label_msgs = [MessageToDict(laser_label_msg) for laser_label_msg in laser_label_msgs]
        laser_labels_json_string = json.dumps(serializable_laser_label_msgs, indent=4, sort_keys=True)
        with open(entry_dir+"laser_labels.json", "w") as file:
            file.write(laser_labels_json_string)

        # load and save projected_lidar_labels
        proj_lidar_labels = [item for item in projected_lidar_labels]
        serializable_proj_lidar_labels = [MessageToDict(proj_lidar_label) for proj_lidar_label in proj_lidar_labels]
        proj_lidar_labels_json_string = json.dumps(serializable_proj_lidar_labels, indent=4, sort_keys=True)
        with open(entry_dir+"projected_lidar_labels.json", "w") as file:
            file.write(proj_lidar_labels_json_string)

        # load and save camera_labels
        camera_label_messages = [item for item in camera_labels]
        serializable_camera_labels = [MessageToDict(camera_label_message) for camera_label_message in camera_label_messages]
        camera_labels_json_string = json.dumps(serializable_camera_labels, indent=4, sort_keys=True)
        with open(entry_dir+"camera_labels.json", "w") as file:
            file.write(camera_labels_json_string)

        # pdb.set_trace()
        # continue

        # load and save images
        for cam_image in frame.images:
            cam_name_str = dataset_pb2.CameraName.Name.Name(cam_image.name)
            image = tf.io.decode_jpeg(cam_image.image).numpy()
            cam_save_path = entry_dir + cam_name_str +".jpg"
            cv2.imwrite(cam_save_path, image)
        # pdb.set_trace()
        # continue

        # load and save lidar
        for laser in frame.lasers:
            laser_name = dataset_pb2.LaserName.Name.DESCRIPTOR.values_by_number[laser.name].name
            # print(laser_name)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser.name)
            # Parse the top laser range image and get the associated projection.
            ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
            # Convert the range image to a point cloud.
            pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
            pcl_with_attr = np.column_stack((pcl, pcl_attr))
            entry_file = laser_name + '.ply'
            save_path = os.path.join(entry_dir, entry_file)
            dataframe = pd.DataFrame(data=pcl_with_attr,
                                    columns=["x", "y", "z",                                    # nlz = "no label zone"
                                            "range", "intensity", "elongation", "is_in_nlz"]) # (1 = in, -1 = not in)
            cloud = PyntCloud(dataframe)
            cloud.to_file(save_path)

        # todo save full pointcloud

        # pdb.set_trace()
        continue

        num_points = 0
        num_features = 7
        point_cloud_list = []

        for laser_id in range(1, 6):

            laser_name = dataset_pb2.LaserName.Name.DESCRIPTOR.values_by_number[laser_id].name

            # Get the laser information
            laser = utils.get(frame.lasers, laser_id)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser_id)

            # Parse the top laser range image and get the associated projection.
            ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

            # Convert the range image to a point cloud.
            pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)

            pcl_with_attr = np.column_stack((pcl, pcl_attr))

            num_points += len(pcl)

            point_cloud_list.append(pcl_with_attr)

            # print(f'{laser_name} LiDAR measured {len(pcl)} points.')

        merged_point_cloud = np.empty((num_points, num_features))

        shift_index = 0

        for point_cloud in point_cloud_list:

            for idx, point in enumerate(point_cloud):

                merged_point_cloud[shift_index + idx] = point

            shift_index += len(point_cloud)

        entry_dir = save_path_root + '/' + str(filename.split('/')[-1]).replace('.tfrecord', '')
        Path(entry_dir).mkdir(exist_ok=True)

        entry_file = f'frame_{str(frame_number).zfill(3)}.ply'

        save_path = os.path.join(entry_dir, entry_file)

        dataframe = pd.DataFrame(data=merged_point_cloud,
                                 columns=["x", "y", "z",                                    # nlz = "no label zone"
                                          "range", "intensity", "elongation", "is_in_nlz"]) # (1 = in, -1 = not in)

        cloud = PyntCloud(dataframe)
        cloud.to_file(save_path)

        frame_number += 1
