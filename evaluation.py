from os.path import join, exists
import numpy as np
import torch
import gym
import gym.envs.box2d
from torchvision import transforms
from models import MDRNNCell, VAE, Controller

# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
ASIZE =3
LSIZE =32
RSIZE =256
RED_SIZE =64
SIZE =64


# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, episodes, device, time_limit):
        """ Build vae, rnn, controller and environment. """

        self.episodes_to_run = episodes
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v0')
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        # print(f"Action: {action.squeeze().cpu().numpy()}")
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=True):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)
        
        episodic_rewards = []
        for i in range(self.episodes_to_run):

            obs = self.env.reset()

            # This first render is required !
            self.env.render()

            hidden = [
                torch.zeros(1, RSIZE).to(self.device)
                for _ in range(2)]

            cumulative = 0
            timestep = 0

            state_rollout = []
            reward_rollout = []
            done_rollout = []
            action_rollout = []

            while True:
                obs = transform(obs).unsqueeze(0).to(self.device)
                action, hidden = self.get_action_and_transition(obs, hidden)
                obs, reward, done, _ = self.env.step(action)

                # state_rollout   += [obs]
                # reward_rollout  += [reward]
                # done_rollout  += [done]
                # action_rollout  += action

                if render:
                    self.env.render()

                cumulative += reward
                if done or timestep > self.time_limit:
                    print(f"Cummulative reward for episode {i}: {cumulative}") #check reward scheme
                    episodic_rewards += [cumulative]
                    break

                    # np.savez(join('evaluation_rollout_{}'.format(i)),
                    #          observations=np.array(state_rollout),
                    #          rewards=np.array(reward_rollout),
                    #          actions=np.array(action_rollout),
                    #          terminals=np.array(done_rollout))

                    # return - cumulative     minuscummulative shows how long stayed in game if not completed
                timestep += 1
        print(f"Episodic rewards: {episodic_rewards}")
        return episodic_rewards
