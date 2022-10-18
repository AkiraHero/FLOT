import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField


class MinimalPublisher(Node):
    def __init__(self):
        rclpy.init(args=None)
        super().__init__('minimal_publisher')
        self.publisher_dict = {}

        
    def register_cloud_topic(self, topic_name):
        self.publisher_dict[topic_name] = self.create_publisher(PointCloud2, topic_name, 10)


    @staticmethod
    def form_a_point_msg(numpy_data):
        # data
        dtype = np.float32
        header = Header(frame_id='map')
        point_data = numpy_data.astype(dtype).tobytes() 
        itemsize = np.dtype(dtype).itemsize
        ros_dtype = PointField.FLOAT32
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]

        msg = PointCloud2(
            header=header,
            height=1, 
            width=numpy_data.shape[0],
            is_dense=False,
            is_bigendian=False,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * numpy_data.shape[0]),
            data=point_data,
            fields=fields,
        )
        return msg
    
    def publish_cloud_data(self, topic, cloud):
        if topic not in self.publisher_dict.keys():
            self.register_cloud_topic(topic)
        msg = self.form_a_point_msg(cloud)
        self.publisher_dict[topic].publish(msg)
        self.get_logger().info('Publishing: "%s"' % topic)
        
        
    def __del__(self):
        self.destroy_node()
        rclpy.shutdown()
        


