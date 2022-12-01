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

    x_T = np.zeros((ui_state.n_samples, H.NOTES, 3), dtype=int)

    x_T[:,:, 0] = 90
    x_T[:, :, 1] = 90
    x_T[:, :, 2] = 512

    x_T[:, :npy.shape[0], 0] = npy[:, 0]
    if npy.shape[1] == 3:
        x_T[:, :npy.shape[0], 1] = npy[:, 1]
        x_T[:, :npy.shape[0], 2] = npy[:, 2]

    audios = sample_audio(x_T)
    update_audio(audios[0], left_audio)

    drawable_samples[0][ui_state.l_selected_index].update_tensor(x_T[0])
    await update_side(0)

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
        ui_state.progress = round(progress, 2)

    def _get_samples():
        return get_samples(ema_sampler, int(ui_state.timesteps), torch.tensor(source, dtype=torch.long).cuda(), progress_handler=progress_handler)

    samples = await io_bound(_get_samples)

    if ui_state.diff_all == 0:
        drawable_samples[side][ui_state.selected_index(side)].update_tensor(samples[0])
    else:
        for i, s in enumerate(samples):
            drawable_samples[side][i].update_tensor(s)

    await update_side(side)


def catch_mouse(e: MouseEventArguments):
    side = 0 if e.sender.view.id == left_pianoroll.view.id else 1
    if e.type == 'mousedown':
        selection_areas[side].open(e.image_x, e.image_y)
    elif e.type == 'mouseup':
        selection_areas[side].close(e.image_x, e.image_y)
        #if selection_areas[side]:
        #    print(selection_areas[side].x, selection_areas[side].y, selection_areas[side].t_x, selection_areas[side].t_y)


async def update_side(side):
    def _fn():
        drawable_samples[side][ui_state.selected_index(side)].draw_melody()
        if side == 0:
            left_pianoroll.source = drawable_samples[side][ui_state.selected_index(side)].to_base64_src_string()
            left_pianoroll.update()
            a = sample_audio(drawable_samples[side][ui_state.selected_index(side)].tensor)
            update_audio(a[0], left_audio)
        else:
            right_pianoroll.source = drawable_samples[side][ui_state.selected_index(side)].to_base64_src_string()
            right_pianoroll.update()
            a = sample_audio(drawable_samples[side][ui_state.selected_index(side)].tensor)
            update_audio(a[0], right_audio)
    await io_bound(_fn)


async def on_mask_click(e: ClickEventArguments):
    side = 0 if e.sender.view.id == left_mask.view.id else 1
    if selection_areas[side]:
        drawable_samples[side][ui_state.selected_index(side)].mask_selection_area(selection_areas[side])
        await update_side(side)


async def diff_handler(e: ClickEventArguments):
    side = 0 if e.sender.view.id == diff_l.view.id else 1
    if drawable_samples[side][ui_state.selected_index(side)].tensor is not None:

        if ui_state.diff_all == 0:
            source = np.expand_dims(drawable_samples[side][ui_state.selected_index(side)].tensor, 0)
        else:
            source = np.array([ds.tensor for ds in drawable_samples[side]])

        await diffuse_to(source, (side + 1) % 2)


async def mask_melody(e: ClickEventArguments):
    side = 0 if e.sender.view.id == lm.view.id else 1

    if drawable_samples[side][ui_state.selected_index(side)].tensor is not None:
        drawable_samples[side][ui_state.selected_index(side)].mask_track(0)
        await update_side(side)


async def mask_bass(e: ClickEventArguments):
    side = 0 if e.sender.view.id == lb.view.id else 1

    if drawable_samples[side][ui_state.selected_index(side)].tensor is not None:
        drawable_samples[side][ui_state.selected_index(side)].mask_track(1)
        await update_side(side)


async def mask_drums(e: ClickEventArguments):
    side = 0 if e.sender.view.id == ld.view.id else 1

    if drawable_samples[side][ui_state.selected_index(side)].tensor is not None:
        drawable_samples[side][ui_state.selected_index(side)].mask_track(2)
        await update_side(side)


async def copy_to(e: ClickEventArguments):
    side = 0 if e.sender.view.id == cl.view.id else 1
    drawable_samples[(side + 1) % 2][ui_state.selected_index((side + 1) % 2)].update_tensor(drawable_samples[side][ui_state.selected_index(side)].tensor)
    await update_side((side + 1) % 2)


async def undo(e: ClickEventArguments):
    side = 0 if e.sender.view.id == lu.view.id else 1
    if drawable_samples[side][ui_state.selected_index(side)].undo():
        await update_side(side)


def on_change_tab(side):
    def _fn(x):
        if left_pianoroll and right_pianoroll:
            asyncio.run_coroutine_threadsafe(update_side(side), asyncio.get_event_loop())
        return x
    return _fn


class UIState:
    n_samples: int = 1
    progress: float = 0.
    audios: list = []
    finished: bool = False
    timesteps: float = 256.
    l_selected_index: int = 0
    r_selected_index: int = 0
    diff_all: int = 0

    def selected_index(self, side):
        if side == 0:
            return self.l_selected_index
        if side == 1:
            return self.r_selected_index


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
        #ui.toggle([2**i for i in range(4)], value=1).bind_value(ui_state, 'n_samples')
        #with ui.row().classes('w-full'):
        ui.slider(value=256, min=100, max=1024 * 3).bind_value_to(ui_state, 'timesteps')
        ui.label().bind_text_from(ui_state, 'timesteps', lambda x: 'timesteps: ' + str(int(x)))

    TABS = 8

    selection_areas = [SelectionArea(), SelectionArea()]
    drawable_samples = [[DrawableSample() for _ in range(TABS)] for _ in range(2)]
    left_pianoroll, right_pianoroll = None, None
    with ui.row():
        with ui.column():
            l_tabs = ui.toggle(list(range(TABS)), value=0).bind_value_to(ui_state, 'l_selected_index', on_change_tab(0))
            left_pianoroll = Pianoroll(on_mouse=catch_mouse, events=['mousedown', 'mouseup'], source=drawable_samples[0][ui_state.l_selected_index].draw_melody().to_base64_src_string())
            left_pianoroll.classes('scrollable-pianoroll')
            with ui.row():
                left_play = ui.button("play").classes("play-left")
                left_stop = ui.button("stop").classes("stop-left")
                left_mask = ui.button("mask", on_click=on_mask_click)
                left_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)
                ui.link('Download MIDI', 'midi/left')

        with ui.column():
            c_tabs = ui.toggle(['s', 'a'], value=0).bind_value_to(ui_state, 'diff_all', lambda x: 0 if x == 's' else 1)
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
            with ui.row():
                lu = ui.button("u", on_click=undo)
                ru = ui.button("u", on_click=undo)

        with ui.column():
            r_tabs = ui.toggle(list(range(TABS)), value=0).bind_value_to(ui_state, 'r_selected_index', on_change_tab(1))
            right_pianoroll = Pianoroll(on_mouse=catch_mouse, events=['mousedown', 'mouseup'], source=drawable_samples[1][ui_state.r_selected_index].draw_melody().to_base64_src_string())
            right_pianoroll.classes('scrollable-pianoroll')
            with ui.row():
                right_play = ui.button("play").classes("play-right")
                right_stop = ui.button("stop").classes("stop-right")
                right_mask = ui.button("mask", on_click=on_mask_click)
                right_audio = ui.html(utils.ui_utils.DUMMY_PLAYER)
                ui.link('Download MIDI', 'midi/right')
    ui.run(reload=False, port=H.port, title='SCHmUBERT')
