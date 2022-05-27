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
from typing import Generic, Iterator, List, Optional, TypeVar

from acme import adders
from acme import core
from acme import specs
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import reverb

Networks = TypeVar('Networks')  # Container for all agent network components.
Policy = TypeVar('Policy')  # Function or container for agent policy functions.
Sample = TypeVar('Sample')  # Generic type for a sample from the replay buffer.


class ActorLearnerBuilder(abc.ABC, Generic[Networks, Policy, Sample]):
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
      policy: Policy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into.

    Args:
      environment_spec: A container for all relevant environment specs.
      policy: Agent's policy which can be used to extract the extras_spec.

    Returns:
      The replay tables used to store the experience the agent uses to train.
    """

  @abc.abstractmethod
  def make_dataset_iterator(
      self,
      replay_client: reverb.Client,
  ) -> Iterator[Sample]:
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
      policy: Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    """Create an actor instance.

    Args:
      random_key: A key for random number generation.
      policy: Instance of a policy expected by the algorithm corresponding to
        this builder.
      environment_spec: A container for all relevant environment specs.
      variable_source: A source providing the necessary actor parameters.
      adder: How data is recorded (e.g. added to replay).
    """

  @abc.abstractmethod
  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: Networks,
      dataset: Iterator[Sample],
      logger: Optional[loggers.Logger],
      environment_spec: Optional[specs.EnvironmentSpec],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """Creates an instance of the learner.

    Args:
      random_key: A key for random number generation.
      networks: struct describing the networks needed by the learner; this can
        be specific to the learner in question.
      dataset: iterator over samples from replay.
      logger: logger used by the constructed learner for logging progress.
      environment_spec: A container for all relevant environment specs.
      replay_client: client which allows communication with replay. Note that
        this is only intended to be used for updating priorities. Samples should
        be obtained from `dataset`.
      counter: a Counter which allows for recording of counts (learner steps,
        actor steps, etc.) distributed throughout the agent.
    """

# TODO(sinopalnikov): deprecated, migrate all users and remove.
GenericActorLearnerBuilder = ActorLearnerBuilder
