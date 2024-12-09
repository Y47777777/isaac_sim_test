#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import threading
import os

rgb_dir = "/home/visionnav/isaac-sim/dataset/rgb/"
depth_dir = "/home/visionnav/isaac-sim/dataset/depth/"
instance_dir = "/home/visionnav/isaac-sim/dataset/instance/"

def rgb_callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    timestr = "%.6f" %  data.header.stamp.to_sec()
            #%.6f表示小数点后带有6位，可根据精确度需要修改；
    image_name = timestr+ ".png" #图像命名：时间戳.jpg
    cv2.imwrite(rgb_dir + image_name, cv_img)  #保存；
    print("save rgb : ", rgb_dir + image_name)
    cv2.imshow("rgb" , cv_img)
    # cv2.waitKey(10)

def depth_callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print(e)
    timestr = "%.6f" %  data.header.stamp.to_sec()
            #%.6f表示小数点后带有6位，可根据精确度需要修改；
    image_name = timestr+ ".png" #图像命名：时间戳.jpg
    cv2.imwrite(depth_dir + image_name, cv_img)  #保存；
    print("save depth : ", depth_dir + image_name)
    cv2.imshow("depth" , cv_img)
    # cv2.waitKey(10)

def instance_callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print(e)
    print(cv_img.max())
    timestr = "%.6f" %  data.header.stamp.to_sec()
            #%.6f表示小数点后带有6位，可根据精确度需要修改；
    image_name = timestr+ ".png" #图像命名：时间戳.jpg
    cv2.imwrite(instance_dir + image_name, cv_img)  #保存；
    print("save instance : ", instance_dir + image_name)
    # cv2.imshow("instance" , cv_img)
    # cv2.waitKey(10)

def odom_callback(data):
    position = data.pose.pose.position
    orientation = data.pose.pose.orientation
    rospy.loginfo("Received odometry data: Position: %s, Orientation: %s", position, orientation)

    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    global bridge
    bridge = CvBridge()

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("rgb", Image, rgb_callback)

    rospy.Subscriber("depth", Image, depth_callback)

    rospy.Subscriber("instance", Image, instance_callback)

    rospy.Subscriber("odom", Odometry, odom_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    for path in [rgb_dir, depth_dir, instance_dir]:
        if not os.path.exists(path):
            os.mkdirs(path)
    listener()
