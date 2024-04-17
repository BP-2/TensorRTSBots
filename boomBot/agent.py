#
# Brady (idea from Jansens greedy bot :) )
#
# BoomBot
# 
# This bot tries to scale its booms
# Seems to do good against potentially slower bots
# Wins majority of long matches
# Gets a little dicey with rush but currently beats RushAI, Frugal, and Jansen's greedy bot (most of the time)
# When there is a rush with a side attack advantage, that is its weakness.
#
# This bot might get screwed with different boom weightings and size mappings tho so watch that

import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation, ActionName, ActionSpace, Action, GlobalCategoricalAction
from TensorRTS import Agent

class boomBot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        self.step_limit = 100
        self.step_count = 1

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}
        self.step_count += 1
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
        
        # Basically, we do a quick check for local resources if nothing good, we boom
        # Then we only rush if someone is close, or if the timer is low
        if self.is_player_one and self.step_limit - self.step_count < 10 and x > 1:
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2]) # We need to rush at end to help ties
        elif self.is_player_two and self.step_limit - self.step_count < 10 and xTwo > 1:
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2]) # We need to rush at end to help ties
        elif self.is_player_one and abs(pos - c_pos) < 4 and dot > 3: # We grab
            if (pos - c_pos) >= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_one and abs(pos - c_pos2) < 4 and dot2 > 3: # We grab
            if (pos - c_pos2) >= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_one and abs(pos - c_pos3) < 4 and dot3 > 3: # We grab
            if (pos - c_pos3) >= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_one and posTwo - pos > 5: # We boom
            mapping["Move"] = GlobalCategoricalAction(3, self.action_space['Move'].index_to_label[3]) # boom     
        elif self.is_player_two and abs(posTwo - c_pos4) < 4 and dot4 > 3: # We grab
            if (posTwo - c_pos4) <= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_two and abs(posTwo - c_pos5) < 4 and dot5 > 3: # We grab
            if (posTwo - c_pos5) <= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_two and abs(posTwo - c_posFinal) < 4 and dotFinal > 3: # We grab
            if (posTwo - c_posFinal) <= 0:
                mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1]) # go back and get it
            else:
                mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        elif self.is_player_two and posTwo - pos > 5: # We boom
            mapping["Move"] = GlobalCategoricalAction(3, self.action_space['Move'].index_to_label[3]) # boom    
        elif self.is_player_two and xTwo > 0:
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2]) # We rush if they are close
        elif self.is_player_one and x > 0:
            mapping["Move"] = GlobalCategoricalAction(2, self.action_space['Move'].index_to_label[2]) # We rush if they are close
        else:
            mapping["Move"] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0]) # go up and get it
        return mapping
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> Agent: 
    return boomBot(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    return 'Boom bot'

if __name__ == "__main__":
  print("\n---Testing Boom Bot---\n")
  from TensorRTS import GameRunner

  cwd = os.getcwd()
  runner = GameRunner(enable_printouts=True, trace_file="BoomTrace.txt")
  init_observation = runner.get_game_observation(is_player_two=False)
  boom_bot = boomBot(init_observation, runner.game.action_space(),cwd)

  runner.assign_players(boom_bot, first_agent_student_name='Boom')
  runner.run(max_game_turns=300)