# from os.path import join, exists
import torch
import gym
import cv2
from models.vae import VAE
from models.mdrnn import MDRNN
from models.controller import Controller
from torchvision.utils import save_image
import gym.envs.box2d

# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64


#Pytorch stuff to seed results and to check for gpu
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(123)

#Constants
latent_size = 32
img_channels = 3
RED_SIZE = 64



# Import models   
# vae_weights = torch.load("exp_dir/vae/best.tar")
# mdrnn_weights = torch.load("exp_dir/mdrnn/best.tar")
# ctrl_weights = torch.load("exp_dir/ctrl/best.tar")




# vae_model = VAE(img_channels, latent_size).to(device)
# vae_model.load_state_dict(vae_weights['state_dict'])
# vae_model.eval()


# 


env = gym.make("CarRacing-v0")
observation = env.reset()

for episode in range(1000):
  env.render()

  action = env.action_space.sample() # your agent [policy(observation)]here (this takes random actions) 
  
  newstate, reward, done, info = env.step(action)
  print(f"observation: {newstate.shape}, action: {action}, reward :: {reward}, done: {done}")

  if done:
    cv2.imwrite(f"data/timestep{episode}.png", newstate)
    observation = env.reset()

