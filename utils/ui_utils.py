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

def add_audio(tensor):
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
    html = ui.html("""
            <audio controls>
                <source type="audio/%s" src="data:audio/%s;base64,%s">
                Your browser does not support the audio tag.
            </audio>
        """ % (mimetype, mimetype, base64.b64encode(bytestr).decode('utf-8')))
    ui.update(html)
    return html


class DrawableSample(object):
    def __init__(self, tensor):
        self.WIDTH = 4096
        self.HEIGHT = 2048
        self.TRACK_OFFSET = 300
        self.scale = 3
        self.tensor = tensor
        self.bitmap = np.ones((self.WIDTH, self.HEIGHT, 3), dtype=np.int8) * 255

    def draw_melody(self, track):
        s = 5
        if track < 2:
            y_off = track * self.TRACK_OFFSET
            for i, pitch in enumerate(self.tensor[:, track]):
                self.bitmap[y_off + pitch * self.scale: y_off + pitch * self.scale + s, i * self.scale: i * self.scale + s] = 0

    def to_src_string(self):
        image = Image.fromarray(self.bitmap.astype(np.int8), mode="RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return "data:image/png;base64," + img_str.decode('utf-8')