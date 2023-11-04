import io

import cv2
import numpy as np
from PIL import Image


def get_img_from_fig(fig, dpi: int = 180) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def allign_arrays(arrs: list[np.ndarray]) -> Image.Image:
    final_shape = (
        sum(arr.shape[0] for arr in arrs),
        max(arr.shape[1] for arr in arrs),
        3,
    )
    img = Image.new("RGB", (final_shape[1], final_shape[0]), (255, 255, 255))
    # center allign, fill with zeros
    h = 0
    for arr in arrs:
        cur_img = Image.fromarray(arr)
        w = (final_shape[1] - arr.shape[1]) // 2
        img.paste(cur_img, (w, h))
        h += arr.shape[0]
    return img
