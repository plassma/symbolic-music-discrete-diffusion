import base64
import os
from pathlib import Path

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
        self.x, self.y = x * 4096, y * 800
        self.t_x, t_y = -1, -1
        self.good = False
    def close(self, x, y):
        self.t_x = x * 4096
        self.t_y = y * 800
        self.good = x > -1 and y > -1
        self.x, self.t_x = sorted([self.x, self.t_x])
        self.y, self.t_y = sorted([self.y, self.t_y])
        return self.good

    def open(self, x, y):
        self.x, self.y = x * 4096, y * 800

    def __bool__(self):
        return self.good
    def __contains__(self, item):
        x, y, w, h = item
        x_contained = self.x <= x and self.t_x >= x + w
        if (h == 0 or w == 0) and self.t_y > 720:
            return x_contained
        return  x_contained and self.y <= y and self.t_y >= y + h


def update_audio(tensor, element):
    import scipy.io.wavfile
    import tempfile
    audiofile = os.path.join(
        tempfile.gettempdir(),
        '%s.wav' % next(tempfile._get_candidate_names()))
    if len(tensor):
        tensor = np.int16(tensor / np.max(np.abs(tensor)) * 32767)
        scipy.io.wavfile.write(audiofile, 44100, tensor)


        extension = audiofile.split('.')[-1].lower()
        mimetypes = {'wav': 'wav', 'mp3': 'mp3', 'ogg': 'ogg', 'flac': 'flac'}
        mimetype = mimetypes.get(extension)
        assert mimetype is not None, 'unknown audio type: %s' % extension
        bytestr = loadfile(audiofile)
        bytestr = base64.b64encode(bytestr).decode('utf-8')
    else:
        bytestr = ""
        mimetype = ""
    element.content = ("""
            <audio controls>
                <source type="audio/%s" src="data:audio/%s;base64,%s">
                Your browser does not support the audio tag.
            </audio>
        """ % (mimetype, mimetype, bytestr))
    element.update()
    return element


def get_styles():
    #all_js_str = [f'<script>{open(f).read()}</script>' for f in list(Path("utils/frontend/MIDIjs/").rglob("*.js"))]
    return open('utils/frontend/dirty_hacks.html').read() + open('utils/frontend/jsmidigen.html').read()# + ''.join(all_js_str)

class DrawableSample():
    def __init__(self, tensor=None):
        if tensor is None:
            tensor = np.ones((1024, 3), int) * (90, 90, 512)
        self.WIDTH = 4096
        self.HEIGHT = 800
        self.DOT_SIZE = 5
        self.DRUMS = 9
        self.scale = 4
        self.TRACK_OFFSET = self.scale * 90
        self.tensors = [tensor]
        self.index = 0
        self.colors = [(255, 0, 0), (0, 0, 255), (0, 0, 0)]

    def get_coords(self, i, track, pitch):
        y_off = (track + 1) * self.TRACK_OFFSET + self.DOT_SIZE * track * 2

        return (i * self.scale), (y_off - pitch * self.scale)

    def undo(self):
        if self.index > 0:
            self.index -= 1
            return True
        return False

    @property
    def tensor(self):
        return self.tensors[self.index]

    def mask_selection_area(self, area: SelectionArea):
        self.next_tensor()
        for track in range(self.tensor.shape[1]):
            s = self.DOT_SIZE
            if track < 2:
                for i, pitch in enumerate(self.tensor[:, track]):
                    x, y = self.get_coords(i, track, pitch)
                    if (x, y, s, s) in area:
                        self.tensor[i, track] = 90
            else:
                drums_tensor = self.tensor[:, track]
                drum_bits = np.array([np.binary_repr(p).zfill(10) for p in drums_tensor])
                for time, drum_tensor in enumerate(drum_bits):
                    if (time * self.scale, 0, 0, 0) in area:
                        self.tensor[time, track] = 512

    def mask_track(self, track):
        self.next_tensor()
        self.tensor[:, track] = 90 if track < 2 else 512

    def update_tensor(self, new_masked_tensor):
        self.next_tensor()
        self.tensors[self.index] = new_masked_tensor.copy()
        self.tensors[self.index][self.tensors[self.index] == 90] = self.tensors[self.index - 1][self.tensors[self.index] == 90]
        self.tensors[self.index][self.tensors[self.index] == 512] = self.tensors[self.index - 1][self.tensors[self.index] == 512]

    def next_tensor(self):
        if len(self.tensors) <= self.index + 1:
            self.tensors.append(None)
        self.index += 1
        self.tensors[self.index] = self.tensors[self.index - 1].copy()