
import multiprocessing as mp
import time


def start_execution():
    queue = mp.Queue(maxsize = 10)
    stop_event = mp.Event()

    queue.put(1)
    queue.put(2)
    queue.put(3)
    queue.put(4)
    queue.put(5)
    queue.put(6)
    queue.put(7)

    processes = [
        mp.Process(target = wait1, args = (queue,)),
        mp.Process(target = wait2, args = (queue,))
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


def wait1(queue):
    while True:
        time.sleep(1)
        print('wait1: ', queue.get())

def wait2(queue):
    while True:
        time.sleep(2)
        print('wait2: ', queue.get())


if __name__ == '__main__':
    start_execution()