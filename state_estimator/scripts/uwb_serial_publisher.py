#!/usr/bin/env python3

# Reads UWB ranging data from a serial port and publishes to /uwb/raw
# Message format expected from device:  "module_id,range_m,self_range_error\n"

import serial
import rospy
from std_msgs.msg import String as StringMsg


def main():
    rospy.init_node("uwb_serial_publisher")

    port  = rospy.get_param("~port",  "/dev/ttyUSB0")
    baud  = rospy.get_param("~baud",  115200)
    topic = rospy.get_param("~topic", "/uwb/raw")

    pub = rospy.Publisher(topic, StringMsg, queue_size=20)

    rospy.loginfo(f"Opening serial port {port} @ {baud} baud")

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        rospy.logerr(f"Failed to open {port}: {e}")
        return

    rospy.loginfo(f"UWB serial publisher ready — publishing on {topic}")

    while not rospy.is_shutdown():
        try:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 3:
                rospy.logwarn_throttle(5, f"Unexpected UWB line: '{line}'")
                continue

            # Validate fields before publishing
            module_id      = int(parts[0])
            range_m        = float(parts[1])
            self_range_err = float(parts[2])

            msg      = StringMsg()
            msg.data = f"{module_id},{range_m},{self_range_err}"
            pub.publish(msg)

        except ValueError as e:
            rospy.logwarn(f"UWB parse error: {e} — line: '{line}'")
        except serial.SerialException as e:
            rospy.logerr(f"Serial read error: {e}")
            break
        except Exception as e:
            rospy.logwarn(f"Unexpected error: {e}")

    ser.close()
    rospy.loginfo("UWB serial publisher shut down.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass