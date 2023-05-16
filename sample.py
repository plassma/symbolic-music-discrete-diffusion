import asyncio
import functools
from typing import Callable

import numpy as np
import torch
from nicegui import ui
from note_seq import midi_to_note_sequence, musicxml_to_sequence_proto
from note_seq import note_sequence_to_midi_file
from starlette import responses

from hparams import get_sampler_hparams
from utils import get_sampler, load_model
from utils.frontend.pianoroll import Pianoroll
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from utils.ui_utils import get_styles


async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))

async def on_upload(e):#todo: could be moved to GUI

    if e.files[0][:4] == b'MThd':
        ns = midi_to_note_sequence(e.files[0])
    elif e.files[0][:19] == b"<?xml version='1.0'":
       ns = musicxml_to_sequence_proto(e.files[0])
    else:
        ui.notify("Seems not to be a valid MIDI or MusicXML file")

    bars = min(64, int(max([n.end_time for n in ns.notes]) // 2))

    npy = ns_to_np(ns, bars, 'trio').outputs

    if len(npy) > 0:
        npy = npy[0]
        ui.notify("interpreting as trio")
    else:
        ui.notify("interpreting as melody")
        npy = ns_to_np(ns, bars, 'melody').outputs[0]
    x_T = np.zeros((ui_state.n_samples, H.NOTES, 3), dtype=int)

    x_T[:, :] = H.codebook_size
    x_T[:, :npy.shape[0], 0] = npy[:, 0]

    if npy.shape[1] == 3:
        x_T[:, :npy.shape[0], 1] = npy[:, 1]
        x_T[:, :npy.shape[0], 2] = npy[:, 2]

    await update_side(0, x_T)


@ui.get('/sgm_plus/{path}')
def get_midi(path: str,):
    print("requested:", path)
    return responses.FileResponse('sgm_plus/' + path, headers={'Access-Control-Allow-Origin': '*'})#

@ui.get('/sgm_plus/{instrument}/{file}')
def get_midi(instrument: str, file: str):
    DEFAULT_VELOCITY = 63
    if not ('_' in file or '.json' in file):
        l = file.split('.')
        file = f'{l[0]}_v{str(DEFAULT_VELOCITY)}.{l[1]}'
    print(f"requested: {instrument}/{file}")
    return responses.FileResponse(f'sgm_plus/{instrument}/{file}', headers={'Access-Control-Allow-Origin': '*'})


async def diffuse_to(source, side):
    def progress_handler(progress):
        ui_state.progress = round(progress / 100, 2)

    def _get_samples():
        return get_samples(sampler, int(ui_state.timesteps), torch.tensor(source, dtype=torch.long).cuda(), progress_handler=progress_handler)

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


def sample_nogui(sampler, H):
    n_samples = 0
    sampler.sampling_batch_size = H.batch_size
    while n_samples < H.n_samples:
        sa = get_samples(sampler, H.sample_steps)
        ns = np_to_ns(sa)

        for n in ns:
            note_sequence_to_midi_file(n, f'data/out/{n_samples}.mid')
            n_samples += 1
        print(f'{n_samples}/{H.n_samples}')


if __name__ == '__main__':
    H = get_sampler_hparams('sample')
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(
                sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)

    if H.no_gui:
        sample_nogui(sampler, H)
    else:
        ui_state = UIState()

        ui.add_head_html(get_styles())

        progress_ui = ui.linear_progress()
        progress_ui.bind_value_from(ui_state, 'progress', lambda x: f'{int(x * 100)}%')

        with ui.expansion('Upload', icon='file_upload').classes('w-full'):
            ui.upload(on_upload=on_upload)
            ui.slider(value=1024, min=1, max=1024 * 3).bind_value_to(ui_state, 'timesteps')
            ui.label().bind_text_from(ui_state, 'timesteps', lambda x: 'timesteps: ' + str(int(x)))

        with ui.row():
            with ui.column():
                left_pianoroll = Pianoroll(H, on_diffuse=on_diffuse)

            with ui.column():
                right_pianoroll = Pianoroll(H, on_diffuse=on_diffuse)

        ui.run(reload=False, port=H.port, title='SCHmUBERT')
