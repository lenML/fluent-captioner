import queue
import threading
import time
import uuid

from cachetools import LRUCache


class Task:
    def __init__(self, task_id, status="waiting", progress=0, result=None, error=None):
        self.task_id = task_id
        self.status = status
        self.progress = progress
        self.result = result
        self.error = error


class TaskQueue:
    def __init__(self, process_task_func, cache_size=128):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.process_task_func = process_task_func
        self.tasks = LRUCache(maxsize=cache_size)
        self.workers = []
        self.running = True

    def add_task(self, **kwargs):
        task_id = str(uuid.uuid4())
        task = Task(task_id)
        with self.lock:
            self.tasks[task_id] = task
            self.queue.put((task_id, kwargs))
        return task_id

    def get_task_status(self, task_id) -> Task:
        with self.lock:
            return self.tasks[task_id]

    def update_task_progress(self, task_id, progress):
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = progress
            else:
                raise ValueError(f"Task {task_id} not found")

    def update_task_status(self, task_id, status, result=None, error=None):
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                task.result = result
                task.error = error
            else:
                raise ValueError(f"Task {task_id} not found")

    def get_queue_status(self):
        with self.lock:
            return {
                "queue_length": self.queue.qsize(),
                "waiting_tasks": len(
                    [t for t in self.tasks.values() if t.status == "waiting"]
                ),
            }

    def start_worker(self, num_workers=1):
        for _ in range(num_workers):
            thread = threading.Thread(target=self.worker)
            thread.start()
            self.workers.append(thread)

    def worker(self):
        while self.running:
            try:
                task_id, kwargs = self.queue.get(timeout=1)
                if task_id is None:
                    break
                self.update_task_status(task_id, "in_progress")
                try:
                    result = self.process_task_func(self, task_id, **kwargs)
                    self.update_task_status(task_id, "completed", result=result)
                except Exception as e:
                    self.update_task_status(task_id, "error", error=str(e))
                    print(f"Error processing task {task_id}: {e}")
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        for worker in self.workers:
            worker.join()
        with self.lock:
            self.tasks.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
            self.queue.task_done()

    def __del__(self):
        self.stop()


# Example usage
if __name__ == "__main__":

    def process_task_example(task_queue, task_id, text=""):
        # Simulate task processing
        for i in range(1, 101):
            if i == 50:  # Simulate an error at 50% progress
                raise ValueError("Simulated error")
            time.sleep(0.1)
            task_queue.update_task_progress(task_id, i)
            print(f"[{text}] progress: {i}%")
        return f"Result of task {task_id}"

    tq = TaskQueue(process_task_example)
    tq.start_worker(num_workers=2)

    task_id1 = tq.add_task(text="Task 1")
    task_id2 = tq.add_task(text="Task 2")

    print(tq.get_task_status(task_id1))
    print(tq.get_queue_status())

    time.sleep(1)
    print(tq.get_task_status(task_id1))

    tq.queue.join()  # Wait for all tasks to be completed

    print(tq.get_queue_status())
    print(tq.get_task_status(task_id1))
    print(tq.get_task_status(task_id2))

    tq.stop()  # Stop the task queue and clear all states
