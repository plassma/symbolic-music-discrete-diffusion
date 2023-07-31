import asyncio
import functools
from typing import Callable

import numpy as np
import torch
from nicegui import ui, app
from note_seq import midi_to_note_sequence, musicxml_to_sequence_proto
from note_seq import note_sequence_to_midi_file
from starlette import responses

from hparams import get_sampler_hparams
from utils import get_sampler, load_model
from utils.frontend.pianoroll import Pianoroll
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from utils.ui_utils import get_styles

LEGEND_SVG = open('utils/frontend/legend.svg', 'r').read()
async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))

@app.get('/soundfonts/{sf_pack}/{path}')#todo: add as static files
def get_midi(sf_pack: str, path: str,):
    print("requested:", path)
    return responses.FileResponse(f'soundfonts/{sf_pack}/' + path, headers={'Access-Control-Allow-Origin': '*'})#

@app.get('/utils/frontend/{script}')
def get_script(script: str):
    return responses.FileResponse(f'utils/frontend/{script}')

@app.get('/soundfonts/{sf_pack}/{instrument}/{file}')
def get_midi(sf_pack: str, instrument: str, file: str):
    DEFAULT_VELOCITY = 85
    if not ('_' in file or '.json' in file):
        l = file.split('.')
        file = f'{l[0]}_v{str(DEFAULT_VELOCITY)}.{l[1]}'
    print(f"requested: {instrument}/{file}")
    return responses.FileResponse(f'soundfonts/{sf_pack}/{instrument}/{file}', headers={'Access-Control-Allow-Origin': '*'})


class UIState:
    n_samples: int = 1
    progress: float = 0.
    finished: bool = False
    timesteps: float = 256.



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


@ui.page('/')
def index():
    ui_state = UIState()

    ui.add_head_html(get_styles())

    async def on_diffuse(event):
        source = np.expand_dims(np.array(event.args['tensor']), 0)
        await diffuse_to(source, (not event.args['direction']) * 1, event.args['target_slot'])

    async def diffuse_to(source, side, target_slot):
        def progress_handler(progress):

            ui_state.progress = round(progress / 100, 2)

        def _get_samples():
            return get_samples(sampler, int(ui_state.timesteps), torch.tensor(source, dtype=torch.long).cuda(),
                               progress_handler=progress_handler)

        samples = await io_bound(_get_samples)

        await update_side(side, samples[0], target_slot)

    async def update_side(side, notes, target_slot=-1):
        def _fn():
            components = [left_pianoroll, right_pianoroll]
            components[side].update_tensor(notes, target_slot)

        await io_bound(_fn)

    async def on_upload(e):  # todo: could be moved to GUI
        bytes = e.content.read()

        if bytes[:4] == b'MThd':
            ns = midi_to_note_sequence(bytes)
        elif bytes[:19] == b"<?xml version='1.0'":
            ns = musicxml_to_sequence_proto(bytes)
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
        x_T = np.zeros((H.NOTES, 3), dtype=int)

        x_T[:] = H.codebook_size
        x_T[:npy.shape[0], 0] = npy[:, 0]

        if npy.shape[1] == 3:
            x_T[:npy.shape[0], 1] = npy[:, 1]
            x_T[:npy.shape[0], 2] = npy[:, 2]

        await update_side(0, x_T)

    progress_ui = ui.linear_progress()
    progress_ui.bind_value_from(ui_state, 'progress', lambda x: f'{int(x * 100)}%')

    with ui.row():
        ui.upload(on_upload=on_upload, label="Upload custom MIDI", auto_upload=True)
        ui.html(LEGEND_SVG)

    ui.slider(value=1024, min=1, max=1024 * 3).bind_value_to(ui_state, 'timesteps')
    ui.label().bind_text_from(ui_state, 'timesteps', lambda x: 'diffusion timesteps: ' + str(int(x)))

    with ui.row():
        left_pianoroll = Pianoroll(props={'mask_ids': H.codebook_size, 'side': 'left'}, on_diffuse=on_diffuse)
        right_pianoroll = Pianoroll(props={'mask_ids': H.codebook_size, 'side': 'right'}, on_diffuse=on_diffuse)

    app.storage.user['count'] = app.storage.user.get('count', 0) + 1
    with ui.row():
        ui.label('your own page visits:')
        ui.label().bind_text_from(app.storage.user, 'count')


if __name__ == '__main__':
    H = get_sampler_hparams('sample')
    H.sample_schedule = "rand"
    sampler = get_sampler(H).cuda()
    sampler = load_model(
                sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)

    if H.no_gui:
        sample_nogui(sampler, H)
    else:

        ui.run(reload=False, port=H.port, title='SCHmUBERT', storage_secret='private key to secure the browser session cookie')