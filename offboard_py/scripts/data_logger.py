#!/usr/bin/env python3

import rospy
import csv
import serial
import threading

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped


class DataLogger:

    def __init__(self):

        rospy.init_node("uwb_imu_gt_logger")

        self.port = rospy.get_param("~port", "/dev/ttyUSB0")
        self.baud = rospy.get_param("~baud", 115200)

        self.imu_file = open("imu.csv", "w", newline="")
        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            "timestamp_sec",
            "ang_vel_x", "ang_vel_y", "ang_vel_z",
            "lin_acc_x", "lin_acc_y", "lin_acc_z",
            "quat_x", "quat_y", "quat_z", "quat_w"
        ])

        self.uwb_file = open("uwb.csv", "w", newline="")
        self.uwb_writer = csv.writer(self.uwb_file)
        self.uwb_writer.writerow([
            "timestamp_sec",
            "module_id",
            "range_m",
            "self_range_error"
        ])

        self.gt_file = open("gt.csv", "w", newline="")
        self.gt_writer = csv.writer(self.gt_file)
        self.gt_writer.writerow([
            "timestamp_sec",
            "gt_x", "gt_y", "gt_z"
        ])

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            rospy.logerr(f"Failed to open serial port {self.port}: {e}")
            self.close()
            raise

        # --- ROS subscribers ---
        rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.gt_callback)

        self.uwb_thread = threading.Thread(target=self.read_uwb, daemon=True)
        self.uwb_thread.start()

        rospy.loginfo("DataLogger initialised — logging IMU, UWB and ground-truth.")

    # Callbacks

    def imu_callback(self, msg):

        timestamp = msg.header.stamp.to_sec()

        self.imu_writer.writerow([
            timestamp,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.imu_file.flush()

    def gt_callback(self, msg):
        
        timestamp = msg.header.stamp.to_sec()

        self.gt_writer.writerow([
            timestamp,
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.gt_file.flush()

    # UWB serial reader

    def read_uwb(self):

        while not rospy.is_shutdown():
            try:
                raw = self.ser.readline()
                line = raw.decode("utf-8", errors="replace").strip()

                if not line:
                    continue

                parts = line.split(",")

                if len(parts) != 3:
                    rospy.logwarn_throttle(5, f"Unexpected UWB line: '{line}'")
                    continue

                module_id      = int(parts[0])
                range_m        = float(parts[1])
                self_range_err = float(parts[2])

                timestamp = rospy.Time.now().to_sec()

                self.uwb_writer.writerow([
                    timestamp,
                    module_id,
                    range_m,
                    self_range_err
                ])
                self.uwb_file.flush()

            except ValueError as e:
                rospy.logwarn(f"UWB parse error: {e} — line was: '{line}'")
            except Exception as e:
                rospy.logwarn(f"UWB read error: {e}")

    # Cleanup

    def close(self):
        for f in (self.imu_file, self.uwb_file, self.gt_file):
            try:
                f.close()
            except Exception:
                pass

        try:
            if self.ser.is_open:
                self.ser.close()
        except AttributeError:
            pass


# Entry point

if __name__ == "__main__":

    logger = DataLogger()

    try:
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    finally:
        logger.close()