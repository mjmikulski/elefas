class HyperPointer:
    def __init__(self, ranges):
        for r in ranges:
            if r < 2: raise ValueError('HyperPointer needs at least two points in each dimension')

        self.ranges = [r-1 for r in ranges]
        self.pos = [0 for _ in ranges]

        self.D = len(ranges) - 1
        self.i = self.D

        self.done = False

    def get(self):
        return tuple(self.pos)

    def move(self):
        while self._is_last_at_this_i():
            self.pos[self.i] = 0
            self.i -=1
            if self.i < 0:
                self.done = True
                return False

        self.pos[self.i] += 1
        self.i = self.D
        return True


    def _is_last_at_this_i(self):
        if self.pos[self.i] == self.ranges[self.i]:
            return True
        else:
            return False
