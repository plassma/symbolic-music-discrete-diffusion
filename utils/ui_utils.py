from io import BytesIO

from PIL import Image
from nicegui import ui
import base64
import os
import numpy as np


def loadfile(filename):
    assert os.path.isfile(filename), 'could not find file %s' % filename
    fileobj = open(filename, 'rb')
    assert fileobj, 'could not open file %s' % filename
    str = fileobj.read()
    fileobj.close()
    return str

DUMMY_PLAYER = """
            <audio controls>
                <source type="audio/wav">
                Your browser does not support the audio tag.
            </audio>
        """

class SelectionArea:
    def __init__(self, x=-1, y=-1):
        self.x, self.y = x, y
        self.t_x, t_y = -1, -1
    def close(self, x, y):
        self.t_x = x
        self.t_y = y
        return x > -1 and y > -1


def update_audio(tensor, element):
    import scipy.io.wavfile  # type: ignore
    import tempfile
    audiofile = os.path.join(
        tempfile.gettempdir(),
        '%s.wav' % next(tempfile._get_candidate_names()))
    tensor = np.int16(tensor / np.max(np.abs(tensor)) * 32767)
    scipy.io.wavfile.write(audiofile, 44100, tensor)


    extension = audiofile.split('.')[-1].lower()
    mimetypes = {'wav': 'wav', 'mp3': 'mp3', 'ogg': 'ogg', 'flac': 'flac'}
    mimetype = mimetypes.get(extension)
    assert mimetype is not None, 'unknown audio type: %s' % extension

    bytestr = loadfile(audiofile)
    element.content = ("""
            <audio controls>
                <source type="audio/%s" src="data:audio/%s;base64,%s">
                Your browser does not support the audio tag.
            </audio>
        """ % (mimetype, mimetype, base64.b64encode(bytestr).decode('utf-8')))
    element.update()
    return element


def get_styles():
    return open('utils/dirty_hacks.html').read()

class DrawableSample():
    def __init__(self, tensor):
        self.WIDTH = 4096
        self.HEIGHT = 800
        self.DRUMS = 9
        self.scale = 4
        self.TRACK_OFFSET = self.scale * 90
        self.tensor = tensor
        self.bitmap = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.int8) * 255

    def draw_melody(self):
        colors = [(255, 0, 0), (0, 0, 255), (0, 0, 0)]
        for track in range(self.tensor.shape[1]):
            s = 5
            y_off = (track + 1) * self.TRACK_OFFSET
            if track < 2:
                for i, pitch in enumerate(self.tensor[:, track]):
                    self.bitmap[y_off - pitch * self.scale - s:y_off - pitch * self.scale, i * self.scale: i * self.scale + s] = colors[track]
            else:
                drums_tensor = self.tensor[:, track]
                drum_bits = np.array([np.binary_repr(p).zfill(10) for p in drums_tensor])

                for time, drum_tensor in enumerate(drum_bits):
                    for i, bit in enumerate(drum_tensor):
                        y_drum = track * self.TRACK_OFFSET + i * s
                        if bit != '0':
                            self.bitmap[y_drum: y_drum + s, time * self.scale: time * self.scale + s] = colors[track]

    def to_base64_src_string(self):
        image = Image.fromarray(self.bitmap.astype(np.int8), mode="RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return "data:image/png;base64," + img_str.decode('utf-8')
