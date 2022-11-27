import asyncio
import base64
import copy
import functools
from asyncio import Task
from io import BytesIO
from threading import Thread
from typing import Callable

import nicegui.globals
import torch
from nicegui.events import MouseEventArguments

import utils.ui_utils
from utils import get_sampler, load_model, EMA
from hparams import get_sampler_hparams
from nicegui import ui
from note_seq import midi_to_note_sequence
from utils.ui_utils import update_audio, DrawableSample, get_styles, SelectionArea
from utils.log_utils import sample_audio
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from PIL import Image
import numpy as np

async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


async def process_midi(e):

    if not e.files[0][:4] == b'MThd' or e.files[0][:19] == b"<?xml version='1.0'":
        ui.notify("Seems not to be a valid MIDI or MusicXML file")

    ns = midi_to_note_sequence(e.files[0])
    npy = ns_to_np(ns, 16).outputs[0]

    drawable_sample = DrawableSample(npy)
    drawable_sample.draw_melody()

    left_ii.source = drawable_sample.to_base64_src_string()
    left_ii.update()

    x_T = np.zeros((ui_state.n_samples, H.NOTES, 3), dtype=np.long)
    x_T[:, :npy.shape[0], 0] = npy[:, 0]

    audios = sample_audio(x_T)
    update_audio(audios[0], left_audio)

    #mask
    x_T[:, :, 1] = 90
    x_T[:, :, 2] = 512
    x_T[:, npy.shape[0]:, 0] = 90

    def progress_handler(progress):
        ui_state.progress = round(progress, 2)

    def _get_samples():
        return get_samples(ema_sampler, int(ui_state.timesteps), torch.tensor(x_T, dtype=torch.long).cuda(), progress_handler=progress_handler)

    samples = await io_bound(_get_samples)
    audios = sample_audio(samples)

    drawable_sample = DrawableSample(samples[0])
    drawable_sample.draw_melody()

    right_ii.source = drawable_sample.to_base64_src_string()
    right_ii.update()

    ui_state.audios = audios

    for a in ui_state.audios:
        update_audio(a, right_audio)

def log_mouse_events(e: MouseEventArguments):
    if e.type == 'mousedown':
        selection_area.x = e.image_x
        selection_area.y = e.image_y
    elif e.type == 'mouseup':
        if selection_area.close(e.image_x, e.image_y):
            print(selection_area.x, selection_area.y, selection_area.t_x, selection_area.t_y)
class UIState:
    n_samples: int = 1
    progress: float = 0.
    audios: list[np.ndarray] = []
    finished: bool = False
    timesteps: float = 1024.


if __name__ == '__main__':
    H = get_sampler_hparams('sample')

    sampler = get_sampler(H).cuda()
    sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    ui_state = UIState()

    ui.add_head_html(get_styles())

    progress_ui = ui.linear_progress()
    progress_ui.bind_value_from(ui_state, 'progress')

    with ui.expansion('Expand!', icon='work').classes('w-full'):
        ui.upload(on_upload=process_midi)
        #ui.toggle([2**i for i in range(4)], value=1).bind_value(ui_state, 'n_samples')
        #with ui.row().classes('w-full'):
        ui.slider(value=256, min=100, max=1024).bind_value_to(ui_state, 'timesteps')
        ui.label().bind_text_from(ui_state, 'timesteps', lambda x: str(int(x)))

    selection_area = SelectionArea()

    with ui.row():
        with ui.column():
            left_ii = ui.interactive_image(on_mouse=log_mouse_events, events=['mousedown', 'mouseup'])
            left_ii.classes('scrollable-pianoroll')
            with ui.row():
                left_play = ui.button("play").classes("play-left")
                left_stop = ui.button("stop").classes("stop-left")
                left_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)
        with ui.column():
            right_ii = ui.interactive_image(on_mouse=log_mouse_events, events=['mousedown', 'mouseup'])
            right_ii.classes('scrollable-pianoroll')
            with ui.row():
                right_play = ui.button("play").classes("play-right")
                right_stop = ui.button("stop").classes("stop-right")
                right_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)

    ui.run(reload=False, port=8081, title='Welcome to SCHmUBERT')

