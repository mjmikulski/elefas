class HyperPointer:
    def __init__(self, ranges):
        for r in ranges:
            if r < 2: raise ValueError('HyperPointer needs at least two points in each dimension')

        self._ranges = [r - 1 for r in ranges]
        self._current_pos = [0 for _ in ranges]

        self._DIM = len(ranges) - 1
        self._index = self._DIM

        self.done = False

    def get(self):
        return tuple(self._current_pos)

    def move(self):
        while self._is_last_at_this_i():
            self._current_pos[self._index] = 0
            self._index -=1
            if self._index < 0:
                self.done = True
                return False

        self._current_pos[self._index] += 1
        self._index = self._DIM
        return True


    def _is_last_at_this_i(self):
        if self._current_pos[self._index] == self._ranges[self._index]:
            return True
        else:
            return False
