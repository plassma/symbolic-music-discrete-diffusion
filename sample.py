import asyncio
import copy
import functools
from typing import Callable

import numpy as np
import torch
from nicegui import ui
from nicegui.events import MouseEventArguments, ClickEventArguments
from note_seq import midi_to_note_sequence, musicxml_to_sequence_proto, note_sequence_to_midi_file
from starlette import responses

from hparams import get_sampler_hparams
from utils import get_sampler, load_model, EMA
from utils.frontend.pianoroll import Pianoroll
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from utils.ui_utils import DrawableSample, get_styles, SelectionArea


async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


def sim_full_instruments():
    s = np.zeros((1024, 3)).astype(int)
    #s[:88, 0] = range(2, 90)
    #s[:88, 1] = range(2, 90)
    s[:9, 2] = [1<<i for i in range(9)]
    ns = np_to_ns(np.expand_dims(s, 0))
    instruments = set([n.instrument for n in ns[0].notes])
    programs = set(n.program for n in ns[0].notes)
    print(instruments, programs)

async def on_upload(e):

    if e.files[0][:4] == b'MThd':
        ns = midi_to_note_sequence(e.files[0])
    elif e.files[0][:19] == b"<?xml version='1.0'":
       ns = musicxml_to_sequence_proto(e.files[0])
    else:
        ui.notify("Seems not to be a valid MIDI or MusicXML file")

    #sim_full_instruments()

    bars = min(64, int(max([n.end_time for n in ns.notes]) // 2))

    npy = ns_to_np(ns, bars, 'trio').outputs

    if len(npy) > 0:
        npy = npy[0]
        ui.notify("interpreting as trio")
    else:
        ui.notify("interpreting as melody")
        npy = ns_to_np(ns, bars, 'melody').outputs[0]

    ns = np_to_ns(np.expand_dims(npy, 0))
    note_sequence_to_midi_file(ns[0], 'test.mid')

    x_T = np.zeros((ui_state.n_samples, H.NOTES, 3), dtype=int)

    x_T[:,:, 0] = 90
    x_T[:, :, 1] = 90
    x_T[:, :, 2] = 512

    x_T[:, :npy.shape[0], 0] = npy[:, 0]
    if npy.shape[1] == 3:
        x_T[:, :npy.shape[0], 1] = npy[:, 1]
        x_T[:, :npy.shape[0], 2] = npy[:, 2]

    await update_side(0, x_T)


@ui.get('/midi/{side}')
def get_midi(side: str,):
    if side == 'left':
        t = drawable_samples[0][ui_state.l_selected_index].tensor.copy()
    else:
        t = drawable_samples[1][ui_state.r_selected_index].tensor.copy()
    t[t == 90] = 0
    t[t == 512] = 0
    ns = np_to_ns(np.expand_dims(t, 0))[0]
    note_sequence_to_midi_file(ns, 'temp.mid')
    return responses.FileResponse('temp.mid', filename=side + '.mid')


async def diffuse_to(source, side):
    def progress_handler(progress):
        ui_state.progress = round(progress / 100, 2)

    def _get_samples():
        return get_samples(ema_sampler, int(ui_state.timesteps), torch.tensor(source, dtype=torch.long).cuda(), progress_handler=progress_handler)

    samples = await io_bound(_get_samples)

    await update_side(side, samples)


async def update_side(side, notes):
    def _fn():
        components = [left_pianoroll, right_pianoroll]
        components[side].notes = notes
        components[side].update()
    await io_bound(_fn)




class UIState:
    n_samples: int = 1
    progress: float = 0.
    finished: bool = False
    timesteps: float = 256.


async def on_diffuse(msg):
    left = msg['id'] == 14
    source = np.array(msg['notes'])
    await diffuse_to(source, left * 1)


if __name__ == '__main__':
    H = get_sampler_hparams('sample')

    sampler = get_sampler(H).cuda()
    sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
    ema = EMA(H.ema_beta)
    ema_sampler = copy.deepcopy(sampler)

    ui_state = UIState()

    ui.add_head_html(get_styles())

    progress_ui = ui.linear_progress()
    progress_ui.bind_value_from(ui_state, 'progress', lambda x: f'{int(x * 100)}%')

    with ui.expansion('Upload', icon='file_upload').classes('w-full'):
        ui.upload(on_upload=on_upload)
        ui.slider(value=256, min=100, max=1024 * 3).bind_value_to(ui_state, 'timesteps')
        ui.label().bind_text_from(ui_state, 'timesteps', lambda x: 'timesteps: ' + str(int(x)))

    with ui.row():
        with ui.column():
            left_pianoroll = Pianoroll(on_diffuse=on_diffuse).classes('scrollable-pianoroll')

        with ui.column():
            right_pianoroll = Pianoroll(on_diffuse=on_diffuse).classes('scrollable-pianoroll')

    ui.run(reload=False, port=H.port, title='SCHmUBERT')
