import sys
from stable_baselines3 import DQN  # Import DQN instead of PPO
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from TensorRTS import TensorRTS_GymEnv
import wandb
from wandb.integration.sb3 import WandbCallback



config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 50000,
}
run = wandb.init(
    project="DQN-Bot",
    tags=["DQN", ],
    config={
        "algorithm": "DQN",
        "timesteps": 50000,
        "policy_type": "MultiInputPolicy",
        "env": "TensorRTS_GymEnv"
    }
)


def custom_win_reward(self):
    if self.is_game_over():
        reward = 0
        if self.has_player_won():
            return 10
        elif self.has_player_lost():
            return 0
        else:
            return 5
    else:
        return 0
    

def custom_power_reward(self, previous_power, current_power):
    power_diff = self.tensor_power(0) - self.tensor_power(1)
    distance = abs(self.tensors[0][0] - self.tensors[1][0])
    if distance == 0:
        return 0
    if power_diff > 0:
        return power_diff - distance
    else:
        return power_diff + distance

def main():
    env = TensorRTS_GymEnv()



    model = DQN.load("dqn_save", env=env)  # Load DQN model if it exists

    model = DQN("MultiInputPolicy", env).learn(  # Use DQN here
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    model.save("dqn_save")

    run.finish()

if __name__ == "__main__":
    main()


# import sys
# from stable_baselines3 import DQN  # Import DQN instead of PPO
# sys.path.append("..")
# sys.path.append("../..")
# sys.path.append("../../..")
# from TensorRTS import TensorRTS_GymEnv
# import wandb
# from wandb.integration.sb3 import WandbCallback
# from stable_baselines3.common.callbacks import BaseCallback




# config = {
#     "policy_type": "MultiInputPolicy",
#     "total_timesteps": 10,
# }
# run = wandb.init(
#     project="DQN-Bot",
#     tags=["DQN", ],
#     config={
#         "algorithm": "DQN",
#         "timesteps": 10,
#         "policy_type": "MultiInputPolicy",
#         "env": "TensorRTS_GymEnv"
#     }
# )


# class WandbLoggingCallback(BaseCallback):

#     def __init__(self, verbose=0):
#         super(WandbLoggingCallback, self).__init__(verbose)
#         self.win_count = 0
#         self.loss_count = 0
#         self.game_count = 0

#     def _on_step(self):
#             infos = self.locals['infos']  # Extracts info for all environments
#             for info in infos:
#                 if 'game_over' in info and info['game_over']:  # Check if a game has ended
#                     self.game_count += 1
#                     if info.get('win', False):  # get win key from info
#                         self.win_count += 1
#                     if info.get('lost', False):  # get lost key from info
#                         self.loss_count += 1
           
            
        
# def custom_win_reward(self):
#     if self.is_game_over():
#         reward = 0
#         if self.has_player_won():
#             return 10
#         elif self.has_player_lost():
#             return 0
#         else:
#             return 5
#     else:
#         return 0
    

# def custom_power_reward(self, previous_power, current_power):
#     power_diff = self.tensor_power(0) - self.tensor_power(1)
#     distance = abs(self.tensors[0][0] - self.tensors[1][0])
#     if distance == 0:
#         return 0
#     if power_diff > 0:
#         return power_diff - distance
#     else:
#         return power_diff + distance

# def main():
#     env = TensorRTS_GymEnv()


#     callback = WandbLoggingCallback()

#     model = DQN.load("dqn_save", env=env)  # Load DQN model if it exists

#     model = DQN("MultiInputPolicy", env).learn(  # Use DQN here
#         total_timesteps=config["total_timesteps"],
#         callback=callback
#     )
#     model.save("dqn_save")
#     win_rate = self.win_count / self.game_count if self.game_count else 0
#     loss_rate = self.loss_count / self.game_count if self.game_count else 0
#     run.log({
#                 'win_rate': win_rate, 
#                 'loss_rate': loss_rate, 
#                 'win_count': self.win_count, 
#                 'loss_count': self.loss_count, 
#                 'games_played': self.game_count
#             })
#     run.finish()

# if __name__ == "__main__":
#     main()
