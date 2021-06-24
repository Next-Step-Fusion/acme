# python3
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

"""RL agent Builder interface."""

import abc
from typing import Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.jax import networks as networks_lib
from acme.utils import counting
import reverb


class ActorLearnerBuilder(abc.ABC):
  """Defines an interface for defining the components of an RL agent.

  Implementations of this interface contain a complete specification of a
  concrete RL agent. An instance of this class can be used to build an
  RL agent which interacts with the environment either locally or in a
  distributed setup.
  """

  @abc.abstractmethod
  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""

  @abc.abstractmethod
  def make_dataset_iterator(
      self,
      replay_client: reverb.Client,
  ) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""

  @abc.abstractmethod
  def make_adder(
      self,
      replay_client: reverb.Client,
  ) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment.

    Args:
      replay_client: Reverb Client which points to the replay server.
    """

  @abc.abstractmethod
  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    """Create an actor instance.

    Args:
      random_key: A key for random number generation.
      policy_network: Instance of a policy network; this should be a callable
        which takes as input observations and returns actions.
      adder: How data is recorded (e.g. added to replay).
      variable_source: A source providing the necessary actor parameters.
    """

  @abc.abstractmethod
  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
      # TODO(mwhoffman): eliminate checkpoint and move it outside.
      checkpoint: bool = False,
  ) -> core.Learner:
    """Creates an instance of the learner.

    Args:
      random_key: A key for random number generation.
      networks: struct describing the networks needed by the learner; this can
        be specific to the learner in question.
      dataset: iterator over samples from replay.
      replay_client: client which allows communication with replay, e.g. in
        order to update priorities.
      counter: a Counter which allows for recording of counts (learner steps,
        actor steps, etc.) distributed throughout the agent.
      checkpoint: bool controlling whether the learner checkpoints itself.
    """
