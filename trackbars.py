import cv2 as cv

class TrackbarBuilder:
    _min = 0
    _max = 1
    _initial = 0
    _name = ""
    _window = ""
    _onchange = None

    def __init__(self, window, name, onChange):
        self._window = window
        self._name = name
        self._onchange = onChange

    def min(self, value):
        self._min = value
        return self

    def max(self, value):
        self._max = value
        return self

    def initial(self, value):
        self._initial = value
        return self

    def build(self):
        cv.createTrackbar(self._name, self._window, 0,
                          self._max, self._onchange)
        cv.setTrackbarMax(self._name, self._window, self._max)
        cv.setTrackbarMin(self._name, self._window, self._min)
        cv.setTrackbarPos(self._name, self._window, self._initial)


class Trackbars:
    window = ""
    onChange = None

    def __init__(self, windowName, onChange):
        self.window = windowName
        self.onChange = onChange

    def add(self, name):
        return TrackbarBuilder(self.window, name, self.onChange)
    
    def addUint8(self, name):
        return self.add(name).min(0).max(255).initial(0)

    def getPos(self, name):
        return cv.getTrackbarPos(name, self.window)

    def setPos(self, name, pos):
        cv.setTrackbarPos(name, self.window, pos)


class ChangeTrackingTrackbars(Trackbars):
    changed = False
    userOnChange = None

    def __init__(self, windowName):
        super().__init__(windowName, self.__onChange)

    def setOnChange(self, method):
        self.userOnChange = method

    def __onChange(self, x):
        self.changed = True
        if self.userOnChange is not None:
          self.userOnChange(x)
