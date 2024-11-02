import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()

def read_depth_color(queue_depth, queue_rgb, stop_event):

    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    print(device)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)

    while not stop_event.is_set():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        if queue_depth.full():
            queue_depth.get()
        queue_depth.put(depth_image)

        if queue_rgb.full():
            queue_rgb.get()
        queue_rgb.put(color_image)


    pipeline.stop()


if __name__ == '__main__':
    pass