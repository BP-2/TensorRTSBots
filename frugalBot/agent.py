#
# Brady (idea from Jansen's greedy bot :) )
#
# FrugalBot
# 
# This bot will move to wherever the biggest amount of coin is 
# If no coin left, it make coin
# Converts its coin when it can
# Does not do big boom (unless there is nothing to grab)
# 
# This bot largely rushes. But it also surveys the ground around it first. Will boom if necessary.
# Seems to split with rush/advance bots and loses to boom bots. 

import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation, ActionName, ActionSpace, Action, GlobalCategoricalAction
from TensorRTS import Agent

class frugalBot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        self.start_position = None

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        # Map elements
        c_pos, dot = current_game_state.features["Cluster"][0]
        c_pos2, dot2 = current_game_state.features["Cluster"][1]
        c_pos3, dot3 = current_game_state.features["Cluster"][2]
        c_pos4, dot4 = current_game_state.features["Cluster"][3]
        c_pos5, dot5 = current_game_state.features["Cluster"][4]
        c_posFinal, dotFinal = current_game_state.features["Cluster"][5]
        
        # Player values
        pos,dim,x,y = current_game_state.features["Tensor"][0] 
        posTwo,dimTwo,xTwo,yTwo = current_game_state.features["Tensor"][1]
        if self.start_position is None:
            self.start_position = pos  # Initialize start position
        if self.is_player_one and pos < 20 and c_pos < pos and dot > 3: # retreat for big dots
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        elif self.is_player_one and pos < 20 and c_pos2 < pos and dot2 > 3: # retreat for big dots
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        elif self.is_player_two and posTwo > 20 and c_posFinal > posTwo and dotFinal > 3: # retreat for big dots
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        elif self.is_player_two and posTwo > 20 and c_pos5 > posTwo and dot5 > 3: # retreat for big dots
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        elif self.is_player_one and x > 1: # Convert the stock to MP
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2])
        elif self.is_player_two and xTwo > 1: # Convert the stock to MP
            # rushin
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2])
        else:
            # advance to get coin
            mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        
        return mapping
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> Agent: 
    return frugalBot(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    return 'Frugal bot'

if __name__ == "__main__":
  print("\n---Testing Frugal Bot---\n")
  from TensorRTS import GameRunner

  cwd = os.getcwd()
  runner = GameRunner(enable_printouts=True, trace_file="FrugalTrace.txt")
  init_observation = runner.get_game_observation(is_player_two=False)
  frugal_bot = frugalBot(init_observation, runner.game.action_space(),cwd)

  runner.assign_players(frugal_bot, first_agent_student_name='Frugal')
  runner.run(max_game_turns=300)