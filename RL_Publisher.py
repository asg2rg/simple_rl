# Input Space
# [distance_to_target, heading_error, obstacle_distance, obstacle_angle]

# Output space
# action: 0 'go_straight'
# action: 1 'turn_left'  
# action: 2 'turn_right'

# subscribe to topics:
# car_x, car_y, car_heading, target_x, target_y (compute distance_to_target and heading_error)
# from lidar: obstacle_x, obstacle_y (compute nearest obstacle distance and angle)

# # publish topics:
# command topic (must convert action space (0, 1, 2) to actual commands for the car)
# linear velocity and angular velocity

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
import numpy as np
import math

class RLCarController(Node):
    def __init__(self):
        super().__init__('rl_car_controller')
        
        # Load trained model
        self.model = PPO.load("simple_rl")  # Path to your trained model
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # State variables
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_heading = 0.0
        self.target_x = 10.0  # Set your target
        self.target_y = 10.0
        self.nearest_obstacle_dist = 100.0
        self.nearest_obstacle_angle = 0.0
        
        # Timer for decision-making
        self.timer = self.create_timer(0.2, self.control_loop)  # 0.2s = dt
        
        self.get_logger().info('RL Car Controller initialized')
    
    def odom_callback(self, msg):
        """Extract position and heading from odometry"""
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        
        # Extract heading from quaternion
        quat = msg.pose.pose.orientation
        self.car_heading = self.quat_to_yaw(quat)
    
    def lidar_callback(self, msg):
        """Process lidar data to find nearest obstacle"""
        ranges = np.array(msg.ranges)
        angles = np.linspace(-msg.angle_min, msg.angle_max, len(ranges))
        
        # Find nearest valid reading
        valid_idx = np.where((ranges > msg.range_min) & (ranges < msg.range_max))[0]
        if len(valid_idx) > 0:
            nearest_idx = valid_idx[np.argmin(ranges[valid_idx])]
            self.nearest_obstacle_dist = ranges[nearest_idx]
            self.nearest_obstacle_angle = angles[nearest_idx]
        else:
            self.nearest_obstacle_dist = 100.0
    
    def control_loop(self):
        """Main control loop - called every 0.2s"""
        # Compute observation (same as training)
        obs = self.compute_observation()
        
        # Get action from trained policy
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to motor command
        cmd = self.action_to_command(action)
        
        # Publish command
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Action: {action}, Linear: {cmd.linear.x}, Angular: {cmd.angular.z}')
    
    def compute_observation(self):
        """Compute 4D observation vector"""
        # Distance to target
        dx = self.target_x - self.car_x
        dy = self.target_y - self.car_y
        distance_to_target = np.sqrt(dx**2 + dy**2)
        
        # Heading error
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.car_heading
        
        # Normalize angle to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        obs = np.array([
            distance_to_target,
            heading_error,
            self.nearest_obstacle_dist,
            self.nearest_obstacle_angle
        ], dtype=np.float32)
        
        return obs
    
    def action_to_command(self, action):
        """Convert discrete action (0,1,2) to Twist command"""
        cmd = Twist()
        
        carvelocity = 20  # m/s (match your training)
        omega = 1.0  # rad/s
        dt = 0.2
        
        if action == 0:  # go_straight
            cmd.linear.x = carvelocity * dt
            cmd.angular.z = 0.0
        elif action == 1:  # turn_left
            cmd.linear.x = carvelocity * 0.6 * dt
            cmd.angular.z = omega
        elif action == 2:  # turn_right
            cmd.linear.x = carvelocity * 0.6 * dt
            cmd.angular.z = -omega
        
        return cmd
    
    def quat_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

def main(args=None):
    rclpy.init(args=args)
    controller = RLCarController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()