# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLFlow logger"""

import os 
import time
from typing import Union, TextIO

import mlflow
import numpy as np

from absl import logging

from acme.utils.loggers import base


MLFLOW_TRACKING_URL = "http://127.0.0.1:5000/" # default value

class MLFlowLogger(base.Logger):
    """An MLFlow logger."""

    def __init__(self, 
                 base_path: Union[str, TextIO] = '~/acme',
                 time_delta: float = 0.,
                 add_uid: bool = True,
                 mlflow_url: str = MLFLOW_TRACKING_URL,
                 ):
        """Instantiates the logger."""

        self._last_log_time = time.time() - time_delta
        self._time_delta = time_delta
        self._add_uid = add_uid
        self._step = 0 # ad-hoc solution to tracking step index

        if "MLFLOW_TRACKING_URI" not in os.environ:
            mlflow.set_tracking_uri(mlflow_url)

    def write(self, data: base.LoggingData):
        """Writes `data` to the MLFlow instance"""

        # Only log if `time_delta` seconds have passed since last logging event.
        now = time.time()

        # TODO(b/192227744): Remove this in favour of filters.TimeFilter.
        elapsed = now - self._last_log_time
        if elapsed < self._time_delta:
            logging.debug('Not due to log for another %.2f seconds, dropping data.',
                            self._time_delta - elapsed)
            return
        self._last_log_time = now

        data = base.to_numpy(data)

        for key, value in data.items():
            # Write numeric data as metrics
            if isinstance(
                value, (int, float, np.float16, np.float32, np.float64)
            ) and not np.isnan(value):
                mlflow.log_metric(key, value, step=self._step)
                del data[key]
            # Maybe it is a better solution to pass `step` in `data` instead
            # but this requires explicit intervention from the calling object
            # Write files as artifacts
            if isinstance(value, os.PathLike):
                mlflow.log_artifact(value,key)
                del data[key]

        # Write the remaining data as a dict.
        mlflow.log_dict(data)

        self._step += 1

    def close(self):
        pass # We expect the MLFlow connection to be closed in the script
