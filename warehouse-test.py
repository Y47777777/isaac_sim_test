import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb.input
import omni.appwindow
import numpy as np
from omni.isaac.core.utils import prims
from omni.isaac.sensor import LidarRtx, Camera, CameraView
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.numpy as numpy_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.wheeled_robots.robots import WheeledRobot
import open3d as o3d
import json
import time

forkilft_prim_path = "/World/forklift"
fork_camera_prim_path = "/World/forklift/lift/fork_camera"
fork_lidar_prim_path = "/World/forklift/lift/fork_lidar"
dataset_dir = "/home/visionnav/isaac-sim/dataset/"
warehouse_asset_path = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
forklift_asset_path = "/Isaac/Robots/Forklift/forklift_c.usd"

# Util function to save rgb annotator data
def write_mormals_data(data, file_path):
    
    print(f"{sys._getframe().f_code.co_name} data.shape: {data.shape}, type: {type(data)}, dtype: {data.dtype}, max: {data.max()}")
    img = Image.fromarray(((data[:,:,:3]  * 0.5 + 0.5)* 255).astype(np.uint8), "RGB")
    img.save(file_path + ".png")


# Util function to save rgb annotator data
def write_rgb_data(data, file_path):
    
    print(f"{sys._getframe().f_code.co_name} data.shape: {data.shape}, type: {type(data)}, dtype: {data.dtype}, max: {data.max()}")
    img = Image.fromarray(data, "RGBA").convert("RGB")
    img.save(file_path + ".png")

# Util function to save depth annotator data
def write_depth_data(data, file_path):
    
    print(f"{sys._getframe().f_code.co_name} data.shape: {data.shape}, type: {type(data)}, dtype: {data.dtype}, max: {data.max()}")
    img = Image.fromarray((data / 10 * 255).astype(np.uint8), "L")
    img.save(file_path + ".png")

# Util function to save semantic segmentation annotator data
def write_sem_data(sem_data, file_path):
    
    # print(sem_data)
    id_to_labels = sem_data["info"]["idToLabels"]
    with open(file_path + ".json", "w") as f:
        json.dump(id_to_labels, f)
    data = sem_data["data"]
    print(f"{sys._getframe().f_code.co_name} data.shape: {data.shape}, type: {type(data)}, dtype: {data.dtype}, max: {data.max()}")
    if sem_data["data"].dtype == np.uint32:
        sem_img = Image.fromarray(data.astype(np.uint8), "L")
        sem_img.save(file_path + ".png")
    elif sem_data["data"].dtype == np.uint8:
        sem_img = Image.fromarray(data, "RGBA").convert("RGB")
        sem_img.save(file_path + ".png")
    
# Util function to save instance segmentation annotator data
def write_inst_data(sem_data, file_path):
    
    # print(sem_data)
    id_to_labels = sem_data["info"]["idToLabels"]
    with open(file_path + ".json", "w") as f:
        json.dump(id_to_labels, f)
    data = sem_data["data"]
    print(f"{sys._getframe().f_code.co_name} data.shape: {data.shape}, type: {type(data)}, dtype: {data.dtype}, max: {data.max()}")
    if sem_data["data"].dtype == np.uint32:
        sem_img = Image.fromarray(data.astype(np.uint16), "I;16")
        sem_img.save(file_path + ".png")
    elif sem_data["data"].dtype == np.uint8:
        sem_img = Image.fromarray(data, "RGBA").convert("RGB")
        sem_img.save(file_path + ".png")
        
def save_camera_data(camera, timestamp):

    camera_current_frame = camera.get_current_frame()
    # rgba
    rgba = camera.get_rgba()
    write_rgb_data(rgba, dataset_dir + str(timestamp) + "_image")
    # depth
    depth = camera.get_depth()
    write_depth_data(depth, dataset_dir + str(timestamp) + "_depth")
    # normals
    normal = camera_current_frame["normals"]
    write_mormals_data(normal, dataset_dir + str(timestamp) + "_normal")
    # instance_segmentation
    inst_seg = camera_current_frame["instance_segmentation"]
    write_inst_data(inst_seg, dataset_dir + str(timestamp) + "_inst_seg")   
    # semantic_segmentation
    sem_seg = camera_current_frame["semantic_segmentation"]
    write_sem_data(sem_seg, dataset_dir + str(timestamp) + "_sem_seg")   
    # pointcloud
    pointcloud = camera.get_pointcloud()
    print(f"pointcloud.shape: {pointcloud.shape}, type: {type(pointcloud)}, dtype: {pointcloud.dtype}")
    np.save(dataset_dir + "camera_pointcloud.npy", pointcloud) 
    # rgb cloud
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud)
    colors = rgba.reshape(-1, 4).astype(np.float64)[:, :3] / 255
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)  
    o3d.io.write_point_cloud(dataset_dir + str(timestamp) + "lidar_point_rgb.pcd", point_cloud_o3d)

def save_lidar_data(lidar, timestamp):
    
    lidar_current_frame = fork_lidar.get_current_frame()
    # print(lidar_current_frame)  
    rendering_time = lidar_current_frame["rendering_time"]
    rendering_frame = lidar_current_frame["rendering_frame"]
    
    point_cloud = lidar_current_frame["point_cloud_data"]
    print(f"{sys._getframe().f_code.co_name} point_cloud.shape: {point_cloud.shape}, type: {type(point_cloud)}, dtype: {point_cloud.dtype}, max: {point_cloud.max()}")
    np.save(dataset_dir + "lidar_pointcloud.npy", point_cloud) 
    
    intensity = lidar_current_frame["intensities_data"]
    print(f"{sys._getframe().f_code.co_name} intensity.shape: {intensity.shape}, type: {type(intensity)}, dtype: {intensity.dtype}, max: {intensity.max()}")
    np.save(dataset_dir + "lidar_intensity.npy", intensity)
     
    elevation = lidar_current_frame["elevation"]
    print(f"{sys._getframe().f_code.co_name} elevation.shape: {elevation.shape}, type: {type(elevation)}, dtype: {elevation.dtype}, max: {elevation.max()}")
    np.save(dataset_dir + "lidar_elevation.npy", elevation) 
    
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    # point_cloud_o3d.colors = o3d.utility.Vector3dVector(np.hstack((intensity.T, intensity.T, intensity.T)))  
    o3d.io.write_point_cloud(dataset_dir + "lidar_point_intensity.pcd", point_cloud_o3d)
    
def forklift_run(forward=True):
    step = 0.1
    if forward == False:
        step = -step
    trans, orint = my_forklift.get_world_pose()
    offset = numpy_utils.quats_to_rot_matrices(quaternions=orint).dot(np.array([0, step, 0], dtype=np.float32)) 
    trans = trans + offset
    my_forklift.set_world_pose(position=trans, orientation=orint) 

def forklift_rotate(forward=True):
    step = 0.1
    if forward == False:
        step = -step
    trans, orint = my_forklift.get_world_pose()
    euler_angles = numpy_utils.quats_to_euler_angles(quaternions=orint)
    euler_angles = euler_angles + np.array([0, 0, step], dtype=np.float32)
    orint = numpy_utils.euler_angles_to_quats(euler_angles=euler_angles)
    my_forklift.set_world_pose(position=trans, orientation=orint)     
    
def lift_move(up=True):
    step = 0.02
    if up == False:
        step = -step
    from omni.isaac.core.prims import XFormPrim
    lift_prim_path = forkilft_prim_path + "/lift"
    # lift_prim = get_prim_at_path(prim_path)
    xform_prim = XFormPrim(prim_path=lift_prim_path)
    trans, orint = xform_prim.get_world_pose()
    print(trans, orint)
    trans = trans + np.array([0, 0, step], dtype=np.float32)
    xform_prim.set_world_pose(position=trans, orientation=orint)
    
# callback
def on_keyboard_event(event):
    
    # print(f"Input event: {event.device} {event.input} {event.keyboard} {event.modifiers} {event.type}")
    # key UP pressed/released
    if event.input == carb.input.KeyboardInput.UP:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            forklift_run(True)
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[20, 0]))
        # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, 0]))
        #     print("Key UP released")
            
    # key DOWN pressed/released
    elif event.input == carb.input.KeyboardInput.DOWN:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            forklift_run(False)
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[-20, 0]))
        # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, 0]))
            
    # key RIGHT pressed/released
    elif event.input == carb.input.KeyboardInput.RIGHT:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            forklift_rotate(False) 
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, 20]))
        # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, 0]))         
            
    # key LEFT pressed/released
    elif event.input == carb.input.KeyboardInput.LEFT:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            forklift_rotate(True) 
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, -20]))
        # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        #     my_forklift.apply_wheel_actions(my_controller.forward(command=[0, 0]))
        
    # key i pressed/released
    elif event.input == carb.input.KeyboardInput.I:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            lift_move(True) 
            
    # key k pressed/released
    elif event.input == carb.input.KeyboardInput.K:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            lift_move(False)     
    
    # key S pressed/released
    elif event.input == carb.input.KeyboardInput.S:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            timestamp = time.time()
            save_camera_data(fork_camera, timestamp)
            save_lidar_data(fork_lidar, timestamp)


            

def add_fork_camera(world):

    width, height = 1280, 960
    camera_matrix = [958.8, 0.0, 957.8, 0.0, 956.7, 589.5, 0.0, 0.0, 1.0]
    distortion_coefficients = [0.05, 0.01, -0.003, -0.0005]
    
    # Camera sensor size and optical path parameters. These parameters are not the part of the
    # OpenCV camera model, but they are nessesary to simulate the depth of field effect.
    #
    # To disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
    pixel_size = 3  # in microns, 3 microns is common
    f_stop = 0.5  # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    focus_distance = 3  # in meters, the distance from the camera to the object plane
    diagonal_fov = 235  # in degrees, the diagonal field of view to be rendered
    
    fork_camera = world.scene.add(
        Camera(prim_path=fork_camera_prim_path, name="fork_camera", 
            translation=np.array([0.0, -80, 20]),
            resolution=(width, height),
            frequency=20,
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True),)
            #    render_product_path=get_active_viewport().get_render_product_path())
    )
    fork_camera.initialize()

    # Calculate the focal length and aperture size from the camera matrix
    (fx, _, cx, _, fy, cy, _, _, _) = camera_matrix
    horizontal_aperture = pixel_size * 1e-3 * width
    vertical_aperture = pixel_size * 1e-3 * height
    focal_length_x = fx * pixel_size * 1e-3
    focal_length_y = fy * pixel_size * 1e-3
    focal_length = (focal_length_x + focal_length_y) / 2  # in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    fork_camera.set_focal_length(focal_length / 10.0)
    fork_camera.set_focus_distance(focus_distance)
    fork_camera.set_lens_aperture(f_stop * 100.0)
    fork_camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    fork_camera.set_vertical_aperture(vertical_aperture / 10.0)
    fork_camera.set_clipping_range(0.05, 1.0e5)

    # Set the distortion coefficients
    # fork_camera.set_projection_type("fisheyePolynomial")
    # fork_camera.set_kannala_brandt_properties(width, height, cx, cy, diagonal_fov, distortion_coefficients)

    # add frame
    fork_camera.add_distance_to_image_plane_to_frame()
    fork_camera.add_distance_to_image_plane_to_frame()
    fork_camera.add_semantic_segmentation_to_frame()
    fork_camera.add_instance_id_segmentation_to_frame()
    fork_camera.add_instance_segmentation_to_frame()
    fork_camera.add_pointcloud_to_frame()
    fork_camera.add_normals_to_frame()
    
    return fork_camera

def add_fork_lidar(world):
    
    fork_lidar = world.scene.add(
        LidarRtx(prim_path=fork_lidar_prim_path, 
                 name="fork_lidar",
                 translation=np.array([0.0, -80, 20]),)
    )
    fork_lidar.add_range_data_to_frame()
    fork_lidar.add_point_cloud_data_to_frame()
    # fork_lidar.enable_visualization()
    fork_lidar.add_intensities_data_to_frame()
    fork_lidar.add_linear_depth_data_to_frame()
    fork_lidar.add_azimuth_range_to_frame()
    fork_lidar.add_horizontal_resolution_to_frame()
    fork_lidar.add_range_data_to_frame()
    fork_lidar.add_azimuth_data_to_frame()
    fork_lidar.add_elevation_data_to_frame()
    
    return fork_lidar
    
def create_forklift(world):
    
    asset_path = assets_root_path + forklift_asset_path
    forklift = world.scene.add(
        WheeledRobot(
            prim_path=forkilft_prim_path,
            name="my_forklift",
            wheel_dof_names=["left_back_wheel_joint", "right_back_wheel_joint"],
            create_robot=True,
            usd_path=asset_path,
            position=np.array([0, 3.0, 0.3]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True)
        )
    )
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    left_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/forklift/left_front_wheel_joint"), "angular")
    left_drive.GetDampingAttr().Set(10000)
    right_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/forklift/right_front_wheel_joint"), "angular")
    right_drive.GetDampingAttr().Set(10000) 
    
    return forklift            

from omni.isaac.core.world import World
import omni.isaac.core.utils.nucleus as nucleus_utils
import sys
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
assets_root_path = nucleus_utils.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

global my_forklift, my_controller
my_forklift = create_forklift(my_world)

from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
my_controller = DifferentialController(name="simple_control", wheel_radius=0.5, wheel_base=1.044)
my_controller.reset()

global fork_lidar, fork_camera, fork_camera_rep
fork_lidar = add_fork_lidar(my_world)
fork_camera = add_fork_camera(my_world)

import omni.isaac.core.utils.stage as stage_utils
warehouse=stage_utils.add_reference_to_stage(
    assets_root_path + warehouse_asset_path, "/background"
)

# get keyboard
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
# subscription
keyboard_event_sub = (carb.input.acquire_input_interface()
                      .subscribe_to_keyboard_events(keyboard, on_keyboard_event))

carb.log_info("running")

my_world.reset()
while simulation_app.is_running():
    my_world.step(render=True)
    fork_camera.get_current_frame()
    fork_lidar.get_current_frame()

my_world.stop()
simulation_app.update()
simulation_app.close()

