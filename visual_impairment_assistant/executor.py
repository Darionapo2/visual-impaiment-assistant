import stream_reader
import multiprocessing as mp
import obstacle_detection
import objects_classification

def start_execution():
    queue_depth = mp.Queue(maxsize = 10)
    queue_rgb = mp.Queue(maxsize = 3)
    queue_obstacles = mp.Queue(maxsize = 10)
    stop_event = mp.Event()

    processes = [
        mp.Process(target = stream_reader.read_depth_color, args = (queue_depth, queue_rgb, stop_event)),
        mp.Process(target = objects_classification.show, args = (queue_rgb, queue_obstacles, stop_event)),
        mp.Process(target = obstacle_detection.show_obstacles, args = (queue_depth, queue_obstacles, queue_rgb, stop_event)),
    ]

    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        stop_event.set()
        for p in processes:
            p.join()
    finally:
        print('Main process exiting')


if __name__ == '__main__':
    start_execution()


