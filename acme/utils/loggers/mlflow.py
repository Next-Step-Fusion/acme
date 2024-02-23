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
from typing import Union, TextIO, Optional

import mlflow
import numpy as np

from absl import logging
from matplotlib.figure import Figure
from mlflow.tracking.client import MlflowClient
from pandas import DataFrame

from acme.utils.loggers import base


MLFLOW_TRACKING_URL = os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:5000/")

class MLFlowLogger(base.Logger):
    """An MLFlow logger."""

    def __init__(self, 
                 label: str = '',
                 time_delta: float = 0.,
                 add_uid: bool = True,
                 mlflow_url: str = MLFLOW_TRACKING_URL,
                 mlflow_run_id: Optional[str] = None,
                 add_step_idx: bool = False,
                 ):
        """Instantiates the logger."""

        self.label = label
        self._run_id = mlflow_run_id or (mlflow.active_run() or mlflow.start_run()).info.run_id
        self._last_log_time = time.time() - time_delta
        self._time_delta = time_delta
        self._add_uid = add_uid
        self._add_step_idx = add_step_idx
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
        unique_fname = f"_{self._step}" if self._add_step_idx else ""
        unique_path = (str(self._step),) if self._add_step_idx else ()

        written_keys = []
        for key, value in data.items():
            # Write numeric data as metrics
            if isinstance(
                value, (int, float, np.float16, np.float32, np.float64)
            ) and not np.isnan(value):
                mlflow.log_metric(self.label+"/"+key, value, step=self._step, run_id=self._run_id)
                written_keys.append(key)
            # Write arrays
            if isinstance(value, np.ndarray): # just assume it is numeric for now
                if value.size==1:
                    mlflow.log_metric(self.label+"/"+key, value.item(), step=self._step, run_id=self._run_id)
                else:
                    # Log metrics as 'key_i_j': value[i,j] etc
                    mlflow.log_metrics({f"{self.label}/{key}_{'_'.join(map(str,idx))}":value[idx]
                                        for idx in np.ndindex(value.shape)}, run_id=self._run_id)
                written_keys.append(key)
            # Write figures
            if isinstance(value, Figure):
                # to be fixed upstream: mlflow.log_figure() does not allow to set run_id
                MlflowClient().log_figure(self._run_id, value, f"{key}{unique_fname}.png")
                written_keys.append(key)
            # Write dataframes
            if isinstance(value, DataFrame):
                mlflow.log_table(value, f"{key}{unique_fname}.json", run_id=self._run_id)
                written_keys.append(key)
            # Write files as artifacts
            if isinstance(value, (str, os.PathLike)): # check for paths 
                if os.path.isfile(value):
                    mlflow.log_artifact(value, os.path.join(self.label,*unique_path,key), run_id=self._run_id)
                    written_keys.append(key)
                elif os.path.isdir(value):
                    mlflow.log_artifacts(value, os.path.join(self.label,*unique_path,key), run_id=self._run_id)
                    written_keys.append(key)                
                    

        # Write the remaining data as a dict.
        # Not sure how fast the list comprehension is here.
        if len(data)>len(written_keys):
            mlflow.log_dict({k:data[k] for k in data if k not in written_keys}, 
                            os.path.join(self.label,f"data{unique_fname}.json"), run_id=self._run_id)

        self._step += 1

    def close(self):
        pass # We expect the MLFlow connection to be closed in the script
