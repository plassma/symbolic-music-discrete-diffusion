import asyncio
import copy
import functools
from typing import Callable

import numpy as np
import torch
from nicegui import ui
from nicegui.events import MouseEventArguments, ClickEventArguments
from note_seq import midi_to_note_sequence, musicxml_to_sequence_proto, note_sequence_to_midi_file
from starlette import responses, requests

import utils.ui_utils
from hparams import get_sampler_hparams
from utils import get_sampler, load_model, EMA
from utils.log_utils import sample_audio
from utils.pianoroll import Pianoroll
from utils.sampler_utils import ns_to_np, get_samples, np_to_ns
from utils.ui_utils import update_audio, DrawableSample, get_styles, SelectionArea


async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


async def on_upload(e):

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

    x_T = np.zeros((ui_state.n_samples, H.NOTES, 3), dtype=np.long)

    x_T[:,:, 0] = 90
    x_T[:, :, 1] = 90
    x_T[:, :, 2] = 512

    x_T[:, :npy.shape[0], 0] = npy[:, 0]
    if npy.shape[1] == 3:
        x_T[:, :npy.shape[0], 1] = npy[:, 1]
        x_T[:, :npy.shape[0], 2] = npy[:, 2]

    audios = sample_audio(x_T)
    update_audio(audios[0], left_audio)

    drawable_samples[0] = DrawableSample(x_T[0])
    drawable_samples[0].draw_melody()

    left_pianoroll.source = drawable_samples[0].to_base64_src_string()
    left_pianoroll.update()

@ui.get('/midi/{side}')
def produce_plain_response(side: str, request: requests.Request):
    if side == 'left':
        t = drawable_samples[0].tensor.copy()
    else:
        t = drawable_samples[1].tensor.copy()
    t[t == 90] = 0
    t[t == 512] = 0
    ns = np_to_ns(np.expand_dims(t, 0))[0]
    note_sequence_to_midi_file(ns, 'temp.mid')
    return responses.FileResponse('temp.mid', filename=side + '.mid')


async def diffuse_to(source, side):
    def progress_handler(progress):
        ui_state.progress = round(progress, 2)

    def _get_samples():
        return get_samples(ema_sampler, int(ui_state.timesteps), torch.tensor(np.expand_dims(source, 0), dtype=torch.long).cuda(), progress_handler=progress_handler)

    samples = await io_bound(_get_samples)
    audios = sample_audio(samples)

    drawable_samples[side] = DrawableSample(samples[0])
    drawable_samples[side].draw_melody()

    if side == 0:
        ii = left_pianoroll
        audio = left_audio
    else:
        ii = right_pianoroll
        audio = right_audio

    ii.source = drawable_samples[side].to_base64_src_string()
    ii.update()

    ui_state.audios = audios

    for a in ui_state.audios:
        update_audio(a, audio)


def catch_mouse(e: MouseEventArguments):
    side = 0 if e.sender.view.id == left_pianoroll.view.id else 1
    if e.type == 'mousedown':
        selection_areas[side].open(e.image_x, e.image_y)
    elif e.type == 'mouseup':
        selection_areas[side].close(e.image_x, e.image_y)
        #if selection_areas[side]:
        #    print(selection_areas[side].x, selection_areas[side].y, selection_areas[side].t_x, selection_areas[side].t_y)


async def update_ii(side):
    def _fn():
        drawable_samples[side].draw_melody()
        if side == 0:
            left_pianoroll.source = drawable_samples[side].to_base64_src_string()
            left_pianoroll.update()
            a = sample_audio(drawable_samples[side].tensor)
            update_audio(a[0], left_audio)
        else:
            right_pianoroll.source = drawable_samples[side].to_base64_src_string()
            right_pianoroll.update()
            a = sample_audio(drawable_samples[side].tensor)
            update_audio(a[0], right_audio)
    await io_bound(_fn)


async def on_mask_click(e: ClickEventArguments):
    side = 0 if e.sender.view.id == left_mask.view.id else 1
    if selection_areas[side]:
        drawable_samples[side].mask_selection_area(selection_areas[side])
        await update_ii(side)


async def diff_handler(e: ClickEventArguments):
    side = 0 if e.sender.view.id == diff_l.view.id else 1
    if drawable_samples[side].tensor is not None:
        await diffuse_to(drawable_samples[side].tensor, (side + 1) % 2)


async def mask_melody(e: ClickEventArguments):
    side = 0 if e.sender.view.id == lm.view.id else 1

    if drawable_samples[side].tensor is not None:
        drawable_samples[side].mask_track(0)
        await update_ii(side)


async def mask_bass(e: ClickEventArguments):
    side = 0 if e.sender.view.id == lb.view.id else 1

    if drawable_samples[side].tensor is not None:
        drawable_samples[side].mask_track(1)
        await update_ii(side)


async def mask_drums(e: ClickEventArguments):
    side = 0 if e.sender.view.id == ld.view.id else 1

    if drawable_samples[side].tensor is not None:
        drawable_samples[side].mask_track(2)
        await update_ii(side)

async def copy_to(e: ClickEventArguments):
    side = 0 if e.sender.view.id == cl.view.id else 1
    old_tensor = drawable_samples[(side + 1) % 2].tensor
    drawable_samples[(side + 1) % 2].tensor = drawable_samples[side].tensor.copy()
    drawable_samples[(side + 1) % 2].tensor[drawable_samples[(side + 1) % 2].tensor == 90] = old_tensor[drawable_samples[(side + 1) % 2].tensor == 90]
    drawable_samples[(side + 1) % 2].tensor[drawable_samples[(side + 1) % 2].tensor == 512] = old_tensor[drawable_samples[(side + 1) % 2].tensor == 512]
    await update_ii((side + 1) % 2)

class UIState:
    n_samples: int = 1
    progress: float = 0.
    audios: list[np.ndarray] = []
    finished: bool = False
    timesteps: float = 256.


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

    with ui.expansion('Upload', icon='file_upload').classes('w-full'):
        ui.upload(on_upload=on_upload)
        #ui.toggle([2**i for i in range(4)], value=1).bind_value(ui_state, 'n_samples')
        #with ui.row().classes('w-full'):
        ui.slider(value=256, min=100, max=1024).bind_value_to(ui_state, 'timesteps')
        ui.label().bind_text_from(ui_state, 'timesteps', lambda x: 'timesteps: ' + str(int(x)))

    selection_areas = [SelectionArea(), SelectionArea()]
    drawable_samples = [DrawableSample(), DrawableSample()]

    with ui.row():
        with ui.column():
            left_pianoroll = Pianoroll(on_mouse=catch_mouse, events=['mousedown', 'mouseup'], source=drawable_samples[0].draw_melody().to_base64_src_string())
            left_pianoroll.classes('scrollable-pianoroll')
            with ui.row():
                left_play = ui.button("play").classes("play-left")
                left_stop = ui.button("stop").classes("stop-left")
                left_mask = ui.button("mask", on_click=on_mask_click)
                left_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)
                ui.link('Download MIDI', 'midi/left')

        with ui.column():
            with ui.row():
                diff_r = ui.button("<=", on_click=diff_handler)
                diff_l = ui.button("=>", on_click=diff_handler)
            with ui.row():
                lm = ui.button("lm", on_click=mask_melody)
                rm = ui.button("rm", on_click=mask_melody)
            with ui.row():
                lb = ui.button("lb", on_click=mask_bass)
                rb = ui.button("rb", on_click=mask_bass)
            with ui.row():
                ld = ui.button("ld", on_click=mask_drums)
                rd = ui.button("rd", on_click=mask_drums)
            with ui.row():
                cr = ui.button("<", on_click=copy_to)
                cl = ui.button(">", on_click=copy_to)

        with ui.column():
            right_pianoroll = Pianoroll(on_mouse=catch_mouse, events=['mousedown', 'mouseup'], source=drawable_samples[1].draw_melody().to_base64_src_string())
            right_pianoroll.classes('scrollable-pianoroll')
            with ui.row():
                right_play = ui.button("play").classes("play-right")
                right_stop = ui.button("stop").classes("stop-right")
                right_mask = ui.button("mask", on_click=on_mask_click)
                right_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)
                ui.link('Download MIDI', 'midi/right')

    ui.run(reload=False, port=8081, title='SCHmUBERT')

