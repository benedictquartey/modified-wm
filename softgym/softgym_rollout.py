import os
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--observation_mode', type=str, default='cam_rgb', help='point_cloud, cam_rgb, key_point')
    parser.add_argument('--rollout_num', type=int, default=1)
    parser.add_argument('--rollout_dir', type=str,  default='data/rollouts')
    # parser.add_argument('--render', action='store_true')
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', action='store_true', help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='data/videos', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')

    args = parser.parse_args()

    # directory stuff
    rollout_dir = args.rollout_dir
    if not os.path.isdir(rollout_dir):
        os.makedirs(rollout_dir)
        print("Created save rollouts directory ...")
    if args.save_video != False:
        if not os.path.isdir(args.save_video_dir):
            os.makedirs(args.save_video_dir)
            print("Created save videos directory ...")


    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['observation_mode'] = args.observation_mode

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))

    for episode in range(args.rollout_num):
        env.reset()
        frames = [env.get_image(args.img_size, args.img_size)]

        timestep = 0
        obs_rollout,r_rollout,a_rollout,d_rollout,i_rollout = [],[],[],[],[]

        for i in range(env.horizon):
            timestep +=1
            action = env.action_space.sample() #random action
            a_rollout.append(action)

            if(args.save_video == False):
                obs, reward, done, info = env.step(action)
            else:
                # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
                # intermediate frames. Only use this option for visualization as it increases computation.
                obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
                frames.extend(info['flex_env_recorded_frames'])

            obs_rollout.append(obs)
            r_rollout.append(reward)
            d_rollout.append(done)
            i_rollout.append(info)

        # Save rollout
        np.savez(os.path.join(rollout_dir, f'{args.env_name}_rollout_{episode}'),
        observations=np.array(obs_rollout),
        rewards=np.array(r_rollout),
        actions=np.array(a_rollout),
        terminals=np.array(d_rollout),
        info=np.array(i_rollout))
        #Save video of rollout
        if args.save_video_dir is not None and args.save_video != False:
            save_name = os.path.join(args.save_video_dir, args.env_name + f'_{episode}.gif')
            save_numpy_as_gif(np.array(frames), save_name)
            print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()