import numpy as np


class ArduinoTalker:

    _max_depth_map = 256
    _max_pwm = 4096
    _encode_const = 4096/(2*_max_depth_map)

    def __init__(self):
        self._port = 80
        self._target_ip = "esp_ip_here"
        self._scale_down = 0.5*ArduinoTalker._encode_const/ArduinoTalker._cube_root(128)

    def _encode(self, _dm_value):
        return int(np.floor(ArduinoTalker._cube_root(_dm_value-128)*self._scale_down) + 0.5*ArduinoTalker._encode_const)

    def encode(self, _arr):
        encoder = np.vectorize(self._encode)
        encoded = encoder(_arr)
        return self._package(encoded)

    @staticmethod
    def _package(_encoded):
        return str.encode("".join(_encoded.reshape(-1).astype(str)) + '9')

    @staticmethod
    def _cube_root(num):
        return num**(1/3) if num > 0 else -abs(num)**(1/3)


if __name__ == '__main__':
    a = ArduinoTalker()
    b = np.array([[255, 128, 55.21108059, 43.85717338, 42.71452991, 43.13849206, 43.47335165],
                  [42.66666667, 43.15537241, 49.74957265, 45.42797619, 44.63899573, 42.78785104, 42.71978022],
                  [42.66666667, 43.15537241, 49.74957265, 45.42797619, 44.63899573, 42.78785104, 0]])
    print(a.encode(b))
