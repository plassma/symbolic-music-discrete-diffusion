from __future__ import annotations

import traceback
from time import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from addict import Dict
from justpy import WebPage
from nicegui.binding import BindSourceMixin, BindableProperty
from nicegui.elements.custom_view import CustomView
from nicegui.elements.element import Element
from nicegui.events import MouseEventArguments
from nicegui.routes import add_dependencies
from nicegui.events import handle_event

from nicegui.binding import bindable_properties, update_views, propagate

add_dependencies(__file__)


class BindableNumpyProperty(BindableProperty):
    def __set__(self, owner: Any, value: Any):
        has_attribute = hasattr(owner, '_' + self.name)
        value_changed = has_attribute and getattr(owner, '_' + self.name) != value
        if isinstance(value_changed, np. ndarray):
            value_changed = value_changed.any()
        if has_attribute and not value_changed:
            return
        setattr(owner, '_' + self.name, value)
        bindable_properties[(id(owner), self.name)] = owner
        update_views(propagate(owner, self.name))
        if value_changed and self.on_change is not None:
            self.on_change(owner, value)


class PianorollView(CustomView):

    def __init__(self, notes: np.ndarray, onDiffuse: Callable, events: List[str]):
        super().__init__('pianoroll', events=events, notes=notes)
        self.allowed_events = ['onConnect', 'onDiffuse']
        self.initialize(onDiffuse=onDiffuse, onConnect=self.on_connect)
        self.sockets = []

    def on_connect(self, msg):
        self.prune_sockets()
        self.sockets.append(msg.websocket)

    def prune_sockets(self):
        page_sockets = [s for page_id in self.pages for s in WebPage.sockets.get(page_id, {}).values()]
        self.sockets = [s for s in self.sockets if s in page_sockets]

_t = 0
def _handle_notes_change(sender: Element, notes) -> None:
    sender.view.options.notes = {'tensor': notes.tolist(), 't': time()}
    sender.update()


class Pianoroll(Element, BindSourceMixin):
    notes = BindableNumpyProperty(on_change=_handle_notes_change)

    def __init__(self, notes: dict = None, *,
                 on_diffuse: Optional[Callable] = None, events: List[str] = ['diffuse']):
        """Interactive Image

        Create an image that handles mouse events and yields image coordinates.
        :param events: list of JavaScript events to subscribe to (default: `['click']`)
        """

        if not isinstance(notes, np.ndarray):
            notes = {'tensor': (np.ones((1024, 3)) * (90, 90, 512)).tolist(), 't': 0}

        self.mouse_handler = on_diffuse
        super().__init__(PianorollView(notes, self.handle_mouse, events))

        self.notes = notes

    def handle_mouse(self, msg: Dict[str, Any]) -> Optional[bool]:
        if self.mouse_handler is None:
            return False
        try:
            return handle_event(self.mouse_handler, msg)
        except:
            traceback.print_exc()
