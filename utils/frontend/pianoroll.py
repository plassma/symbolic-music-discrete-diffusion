from __future__ import annotations

from time import time
from typing import Callable, Optional

import numpy as np
from nicegui.element import Element


class Pianoroll(Element, component='pianoroll.js'):
    def __init__(self, props: dict = None, *,
                 on_diffuse: Optional[Callable] = None):
        """Interactive Image

        Create an image that handles mouse events and yields image coordinates.
        :param events: list of JavaScript events to subscribe to (default: `['click']`)
        """
        if not 'mask_ids' in props.keys():
            raise Exception('You must at least provide mask ids')
        if not 'tensor' in props.keys():
            props['tensor'] = (np.ones((1024, 3)) * np.array(props['mask_ids'])).tolist()
        if not 't' in props.keys():
            props['t'] = 0
        if not 'side' in props.keys():
            props['side'] = 'left'

        super().__init__()
        super().on('diffuse', on_diffuse)
        super().classes('scrollable-pianoroll')

        self._props |= props
        self._props['id'] = self.id

    def update_tensor(self, tensor, target_slot=-1):
        self._props['tensor'] = tensor
        self._props['t'] = time()
        self._props['target_slot'] = target_slot
        super().update()
