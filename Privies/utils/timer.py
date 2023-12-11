import time
"""
func : 记录时间
start_time
average_time
total_time
duration 两次记录中间的输出结果
_calls: 记录记录时间

"""

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = time.time() # 初始化起始时间
        # self.diff = 0.
        self.average_time = 0.
        self._last_time = self.start_time
        self.duration = 0.

    """记录开始时间"""
    def start_record(self):
        self.start_time = time.time()

    """ 记录时间，如果输出均值时间则average = True"""
    def record(self, average=False):
        diff = time.time() - self._last_time
        self._last_time = time.time()
        self.total_time = self._last_time - self.start_time
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.average_time = 0.
        self.duration = 0.

"""测试"""
if __name__ == "__main__":
    timer = Timer()
    time.sleep(1)
    print("{:.4f}".format(timer.record(False)))
    time.sleep(2)
    print("{:.4f}".format(timer.record(False)))
    time.sleep(3)
    print("{:.4f}".format(timer.record(False)))
    time.sleep(10)
    print("{:.4f}".format(timer.record(False)))
    