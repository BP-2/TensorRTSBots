import random
from typing import Dict, Mapping
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent
from stable_baselines3 import DQN  # Import DQN instead of PPO

class Feynman_Agent(Agent):
    def __init__(self, init_observation: Observation, action_space: Dict[ActionName, ActionSpace], script_dir: str) -> None:
        super().__init__(init_observation, action_space, script_dir)

    def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        action_choice = random.randrange(0, 2)
        if action_choice == 1:
            mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        else:
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])

        return mapping

    def on_game_start(self, is_player_one: bool, is_player_two: bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)

    def on_game_over(self, did_i_win: bool, did_i_tie: bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)

def agent_hook(init_observation: Observation, action_space: Dict[ActionName, ActionSpace], script_dir: str) -> Agent:
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    return Feynman_Agent(init_observation, action_space, script_dir)

def display_name_hook() -> str:
    """Provide the name to refer to the agent as a string

    Returns:
        str: Name to display
    """
    return 'feynmanDQN'
