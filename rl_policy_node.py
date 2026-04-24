# Input Space
# [distance_to_target, heading_error, obstacle_distance, obstacle_angle]

# Output space
# action: 0 'go_straight'
# action: 1 'turn_left'  
# action: 2 'turn_right'

# subscribe to topics:
# bot_x, bot_y, bot_yaw, target_x, target_y (compute distance_to_target and heading_error)
# from lidar: obstacle_x, obstacle_y (compute nearest obstacle distance and angle)

# # publish topics:
# command topic (must convert action space (0, 1, 2) to actual commands for the car)
# linear velocity and angular velocity

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Point
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
import numpy as np
import math


class RLCarController(Node):
    def __init__(self):
        super().__init__("rl_car_controller")

        # Load the RL model
        self.model = PPO.load("/home/jinhong/models/model_PPO29_Best.zip") # Is this the right way to load the model?

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmds", 10)

        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Vector3,'/true_robot_position_oriention', self.position_orientation_callback,10)
        self.velocity_sub = self.create_subscription(Twist, '/position_velocity_orientation', self.velocities_callback, 10)
        self.rabbit_sub = self.create_subscription(Point, 'rabbit_location', self.rabbit_callback, 10)


        # Robot state
        self.bot_x = None
        self.bot_y = None
        self.bot_yaw = None

        # Robot velocity
        self.prev_linear_velocity = 0.0
        self.prev_angular_velocity = 0.0

        # Target state
        self.target_x = None
        self.target_y = None

        # Obstacle state
        self.obstacle_distance = 100.0
        self.obstacle_angle = 0.0

        self.policy_timer = self.create_timer(0.2, self.policy_process)

        self.get_logger().info("RL controller initialized.")

    def position_orientation_callback(self, msg):
        self.bot_x = msg.x
        self.bot_y = msg.y
        self.bot_yaw = math.radians(msg.z)
        self.bot_yaw = self.wrap_angle(self.bot_yaw) # force it to be between -pi and pi

    def velocities_callback(self, msg):
        self.prev_linear_velocity = msg.linear.x
        self.prev_angular_velocity = math.radians(msg.angular.x)
    

    def rabbit_callback(self, msg):
        if msg is None:
            return
        self.target_x = msg.x
        self.target_y = msg.y

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        valid = np.isfinite(ranges)
        valid = valid & (ranges > msg.range_min)
        valid = valid & (ranges < msg.range_max)

        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            self.obstacle_distance = 100.0 
            self.obstacle_angle = 0.0
            return

        nearest_index = valid_indices[np.argmin(ranges[valid_indices])]

        nearest_distance_m = float(ranges[nearest_index])
        nearest_angle = msg.angle_min + nearest_index * msg.angle_increment

        self.obstacle_distance = nearest_distance_m * 100.0 # Make the scale look similar to the RL policy

        if self.obstacle_distance > 100.0:
            self.obstacle_distance = 100.0

        self.obstacle_angle = self.wrap_angle(nearest_angle)

    def policy_process(self):
        if not self.has_all_data():
            return

        obs = self.compute_observation()

        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        cmd = self.action_to_command(action)

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"obs={obs}, action={action}, "
            f"linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}"
        )

    def has_all_data(self):
        if self.bot_x is None:
            self.get_logger().warn("Missing bot_x")
            return False
        if self.bot_y is None:
            self.get_logger().warn("Missing bot_y")
            return False
        if self.bot_yaw is None:
            self.get_logger().warn("Missing bot_yaw")
            return False
        if self.target_x is None:
            self.get_logger().warn("Missing target_x")
            return False
        if self.target_y is None:
            self.get_logger().warn("Missing target_y")
            return False
        return True

    # match the observation space defined in the RL policy
    def compute_observation(self):
        dx = self.target_x - self.bot_x
        dy = self.target_y - self.bot_y

        distance_to_target = math.sqrt(dx * dx + dy * dy)

        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.bot_yaw
        heading_error = self.wrap_angle(heading_error)

        obs = np.array([distance_to_target, heading_error, self.obstacle_distance, self.obstacle_angle,], dtype=np.float32,)
        return obs

    #convert the action (0, 1, 2) to actual commands for the car
    def action_to_command(self, action):
        cmd = Twist()

        if action == 0: # go straight
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
        elif action == 1: # turn left
            cmd.linear.x = 0.3
            cmd.angular.z = 0.8
        elif action == 2: # turn right
            cmd.linear.x = 0.3
            cmd.angular.z = -0.8
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def wrap_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = RLCarController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()