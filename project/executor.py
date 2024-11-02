from concurrent.futures import ProcessPoolExecutor
import time
import main
import natual_language_compiler

# the global executor
executor = ProcessPoolExecutor(max_workers = 2)

def execution():
    res = []

    main_task = executor.submit(main.open_cv_viewer)

    # wait for the tasks to complete
    res.append(main_task.result())

    return res

if __name__ == '__main__':
    execution()