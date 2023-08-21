import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from tensorflow.python.training.summary_io import summary_iterator

keys = [
        'ray/tune/evaluation/episode_reward_mean',
        # 'ray/tune/iterations_since_restore'
        ]

def extract_data_from_events(path, tags):
    data = {key : [] for key in tags+['step']}
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag in tags :
                data.get('step').append(int(e.step))
                data.get(v.tag).append(v.simple_value)

    return pd.DataFrame.from_dict(data)

def extract_data_from_events_v2(path, tags):
    """
    here, total_df is a smaller df with possible values in one column, whose index is step.
    """
    data = {key : {'step': [], key: []} for key in tags}
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag in tags :
                data_dict = data.get(v.tag)
                data_dict.get('step').append(int(e.step))
                data_dict.get(v.tag).append(v.simple_value)
    total_df = pd.DataFrame()
    for key, val in data.items():
        df = pd.DataFrame.from_dict(val)
        df = df.set_index('step')
        total_df = total_df.join(df, how='outer')

    return total_df

def plot(data_source = 'events'):
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])
    # path_dict = {
    #     '4096': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_19-46-56wxlo9bqe',
    #     '8192': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_09-03-02ajit44mi',
    #     '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_09-18-428jl_v2ly',
    #     '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    # } #sin

    # path_dict = {
    #     '4096': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_19-46-56lko7ykvm',
    #     '8192': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_20-57-36ptuduzud',
    #     '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_20-57-36xmnfv1f7',
    #     '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    # } # theta directly

    # Cartpole

    # path_dict = {
    #     '2048exp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-51-176qhk6ywr',
    #     '4096exp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-51-177n68vpde',
    #     '8192exp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-51-17fie8olrt',
    #     'SACsinthexp' : '/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-51-17i_cw3m6f',
    #     '8192': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-02-30x2m7zjo4',
    #     '4096': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-02-3013hj5ilk',
    #     '2048': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-02-30sn_cqvve',
    #     'SACsinth': '/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-02-30fqar1_os',
    #     '8192th': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-21nv70ribt',
    #     '4096th': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-21lnb8dai3',
    #     '2048th': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-215bpvmwd3',
    #     'SACth': '/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-37-2148n_4kxq',
    #     '8192thexp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_09-14-10_dpm46gh',
    #     '4096thexp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_09-14-10bptykb5h',
    #     '2048thexp': '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_09-14-10kwro4fll',
    #     'SACthexp': '/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_09-14-10dwii3is2',
    #
    # } # theta directly

    # Pendubot
    # path_dict = {
    #     r'$r_{energy}$' : '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-19_10-37-28nljvhenj',
    #     r'$r_{q}$' : '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-15_23-57-14kx2hy4ke'
    # }

    path_dict = {

        'Random feature'    : ['/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-16_00-59-52isfu35bc'],
        'SAC'               : ['/home/mht/ray_results/SAC_Pendubot-v0_2023-08-16_00-34-47hsa2t7v3'],
        'Nystrom'           : [
                                # '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-16_02-36-2422vs088c',
                                '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-16_08-07-26l8n5k1pj',
                              ],
        'Nystrom_top1024'   : ['/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-21_04-37-17wehtdvy7'],
        'Nystrom_top_512'   : ['/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-21_00-27-29xac48rwj']
    }

    # Pendulum

    # path_dict = {
    #     'Nystrom_Top256_2': [
    #         # '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-08-16_02-36-2422vs088c',
    #         '/media/mht/新加卷/ray_results/Pendulum/RFSAC_Pendulum-v1_2023-08-20_22-00-26v2j5cbu2',
    #     ],
    #     'Nystrom_512': ['/media/mht/新加卷/ray_results/Pendulum/RFSAC_Pendulum-v1_2023-08-20_20-38-03x328v7kl'],
    #     'Nystrom_Top256': ['/media/mht/新加卷/ray_results/Pendulum/RFSAC_Pendulum-v1_2023-08-20_18-59-203xw8qkck'],
    #     'SAC': ['/home/mht/ray_results/SAC_Pendulum-v1_2023-08-21_02-06-12hdx3gpw_'],
    #     'random_feature':['/home/mht/ray_results/RFSAC_Pendulum-v1_2023-08-21_01-40-54r4vmpr25']
    # }

    total_iters = 1000.
    dfs = []
    for key, paths in path_dict.items():
        # rfdim = int(rfdim)
        for path in paths:
            if data_source == 'csv':
                df = pd.read_csv(os.path.join(path, 'progress.csv'))
            elif data_source == 'events':
                for fname in os.listdir(path):
                    if fname.startswith('events'):
                        break
                df = extract_data_from_events(os.path.join(path, fname), keys)

            df['Algorithm'] = key
            # normalize potentially different steps per iteration.
            max_step = df['step'].values.max()
            step_to_iter_factor = max_step / total_iters
            df['training_iteration'] = df['step'] / step_to_iter_factor

            # df['episode_reward_evaluated'] = np.log(df['episode_reward_mean'] / 200.) / 10. * 200
            # if rfdim.startswith('SAC'):
            #     df['exp_setup'] = rfdim
            # elif rfdim.endswith('thexp'):
            #     df['exp_setup'] = 'thexp'
            # elif rfdim.endswith('exp'):
            #     df['exp_setup'] = 'sinthexp'
            # elif rfdim.endswith('th'):
            #     df['exp_setup'] = 'th'
            # else:
            #     df['exp_setup'] = 'sinth'
            dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in keys: # 'episode_reward_min', 'episode_reward_max',
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='Algorithm', palette='muted')
        # plt.tight_layout()
        title = y.split('/')[-1] + 'Pendubot'
        plt.title(title)
        plt.ylabel('')
        # plt.xlim(0, 800)
        # plt.ylim(-1000, 20)
        plt.xlabel('training iterations')
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/rllib-random-feature/fig/' + title + '.png'
        plt.savefig(figpath)

def plot_pendulum():
    sns.set(style='darkgrid', font_scale=1)
    sns.set_palette([(0.0, 0.24705882352941178, 1.0),
                     (0.011764705882352941, 0.9294117647058824, 0.22745098039215686),
                     (0.9098039215686274, 0.0, 0.043137254901960784),
                     (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
                     (1.0, 0.7686274509803922, 0.0),
                     (0.0, 0.8431372549019608, 1.0)])
    path_dict = {
        'Random Feature SAC': '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-24_09-36-31jrzl7hdp',
        'SAC' : '/home/mht/ray_results/SAC_Pendulum-v1_2023-04-23_19-18-33qzefa_7_',
        # '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_09-18-428jl_v2ly',
        # '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    }

    dfs = []
    for rfdim, path in path_dict.items():
        # rfdim = int(rfdim)
        df = pd.read_csv(os.path.join(path, 'progress.csv'))
        df['algorithm'] = rfdim
        a = 0
        dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in ['episode_reward_mean', ]: # 'episode_reward_min', 'episode_reward_max', 'episode_len_mean'
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='algorithm', palette='muted')
        plt.tight_layout()
        plt.xlim([-2, 500])
        plt.title('Mean episodic return')
        plt.ylabel('')
        plt.xlabel('training iterations')
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/rllib_random_feature/fig/pen_' + y + '.png'
        plt.savefig(figpath)

if __name__ == '__main__':
    plot()
