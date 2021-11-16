import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--rollout_num', type=int, default=1)
    parser.add_argument('--rollout_dir', type=str,  default='./data/')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')

    args = parser.parse_args()
    rollout_dir = args.rollout_dir

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = args.render
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))

    for episode in range(args.rollout_num):
        env.reset()
        frames = [env.get_image(args.img_size, args.img_size)]

        timestep = 0
        obs_rollout,r_rollout,a_rollout,d_rollout,i_rollout = [],[],[],[]

        for i in range(env.horizon):
            timestep +=1
            action = env.action_space.sample() #random action
            # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
            # intermediate frames. Only use this option for visualization as it increases computation.
            if(arg.save_video == False):
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
                frames.extend(info['flex_env_recorded_frames'])
            
        # Save rollout
        np.savez(osp.join(rollout_dir, 'rollout_{}'.format(i)),
        observations=np.array(obs_rollout),
        rewards=np.array(r_rollout),
        actions=np.array(a_rollout),
        terminals=np.array(d_rollout),
        info=np.array(i_rollout))
        #Save video of rollout
        if args.save_video_dir is not None and args.save_video != False:
            save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
            save_numpy_as_gif(np.array(frames), save_name)
            print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
