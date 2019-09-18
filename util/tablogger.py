import numpy as np
import matplotlib.pyplot as plt


class TabLogger(object):
    """Save scalars to log file, simple functionality to plot values"""
    COL_SEP = "\t"
    ROW_SEP = "\n"

    def __init__(self, fpath, resume=False):
        """Initialize a file at fpath
        """
        self.resume = resume
        if resume:
            self.file = open(fpath, "r")
            name = self.file.readline()
            self.names = name.rstrip().split(TabLogger.COL_SEP)
            self.numbers = {}
            for name in self.names:
                self.numbers[name] = []
            for numbers in self.file:
                numbers = numbers.rstrip().split(TabLogger.COL_SEP)
                for idx, number in enumerate(numbers):
                    self.numbers[self.names[idx]].append(number)
            self.file.close()
            self.file = open(fpath, "a")
        else:
            self.file = open(fpath, "w")

    def set_names(self, names, flush=True):
        if self.resume:
            pass
        self.numbers = {}
        self.names = names
        for name in self.names:
            self.file.write(name)
            self.file.write(TabLogger.COL_SEP)
            self.numbers[name] = []
        self.file.write(TabLogger.ROW_SEP)
        if flush:
            self.file.flush()

    def append(self, numbers, flush=True):
        assert len(self.names) == len(self.numbers), "numbers length must match names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write(TabLogger.COL_SEP)
            self.numbers[self.names[index]].append(num)
        self.file.write(TabLogger.ROW_SEP)
        if flush:
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()

    def plot(self):
        for name in self.names:
            x = np.arrange(len(self.numbers[name]))
            plt.plot(x, np.asarray(self.numbers[name]))
        plt.legend(self.names)
        plt.grid(True)
