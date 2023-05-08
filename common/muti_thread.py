#!/usr/bin/python3

import queue
import threading


class ProcessThread(threading.Thread):

    def __init__(self, thread_id, queue, queue_lock):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.queue = queue
        self.queue_lock = queue_lock
        self.result = []
        # 通知线程是时候退出

    def process(data):
        pass

    def run(self, data):
        while not self.queue.empty():
            self.queue_lock.acquire()
            if not self.queue.empty():
                data = self.queue.get()
                self.process(data)
                # todo
                # process(data)
                self.result.extend(rs)
                self.queueLock.release()
            else:
                self.queueLock.release()


class MutiThread:

    def __init__(self, queue_size) -> None:
        self.queue_lock = threading.Lock()
        self.work_queue = queue.Queue(queue_size)

    def bootstrap(self, data, threads):

        # 填充队列
        self.queue_lock.acquire()
        for d in data:
            self.work_queue.put(d)
        self.queue_lock.release()

        # 启动线程
        for t in threads:
            t.start()

        # 等待队列清空
        while not self.work_queue.empty():
            pass

        # 等待所有线程完成
        result = []
        for t in threads:
            t.join()
            thread_result = t.result
            result.extend(thread_result)
        return result


if __name__ == "__main__":

    def deal(data):
        print(data)
        return [data]

    mt = MutiThread(thread_num=2, queue_size=10)
    data = ["sssssss", "eeeeeeee", "eeeeeeooooo"]
    rs = mt.bootstrap(data)
    # 填充队列
    # mt.queueLock.acquire()
    # for d in data:
    #     mt.workQueue.put(d)
    # mt.queueLock.release()
