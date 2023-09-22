import os

import clearml
import numpy as np

task = clearml.Task.init("afgsgfd", "dfagagafb")
logger = task.logger
m = np.eye(256, 256, dtype=float)
logger.report_image("image", "image float", iteration=1, image=m)
# print(os.environ)
