import numpy as np


class ArduinoTalker:

    _max_depth_map = 256
    _max_pwm = 4096
    _encode_const = 2

    def __init__(self):
        self._port = 80
        self._target_ip = "esp_ip_here"
        self._scale_down = ArduinoTalker._max_pwm/(ArduinoTalker._encode_const*(ArduinoTalker._max_depth_map ** 2))

    def _encode(self, _dm_value):
        return int(np.floor(_dm_value*self._scale_down))

    def encode(self, _arr):
        encoder = np.vectorize(self._encode)
        encoded = encoder(_arr)
        return self._package(encoded)

    @staticmethod
    def _package(_encoded):
        return str.encode("".join(_encoded.reshape(-1).astype(str)))


if __name__ == '__main__':
    a = ArduinoTalker()
    b = np.array([[42.66666667, 46.39310134, 55.21108059, 43.85717338, 42.71452991, 43.13849206, 43.47335165],
         [42.66666667, 43.15537241, 49.74957265, 45.42797619, 44.63899573, 42.78785104, 42.71978022],
             [42.66666667, 43.15537241, 49.74957265, 45.42797619, 44.63899573, 42.78785104, 42.71978022]])
    print(a.encode(b))
