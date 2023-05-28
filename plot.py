import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot():
    sns.set(style='darkgrid', font_scale=1.3)
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

    path_dict = {
        '4096': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_19-46-56lko7ykvm',
        '8192': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_20-57-36ptuduzud',
        '16384': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_20-57-36xmnfv1f7',
        '32768': '/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt'
    } # theta directly

    dfs = []
    for rfdim, path in path_dict.items():
        rfdim = int(rfdim)
        df = pd.read_csv(os.path.join(path, 'progress.csv'))
        df['random_feature_dim'] = rfdim
        a = 0
        dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    for y in ['episode_reward_mean', 'episode_reward_min', 'episode_reward_max', 'episode_len_mean']:
        plt.figure(figsize=[6, 4])
        sns.lineplot(total_df, x='training_iteration', y=y, hue='random_feature_dim', palette='muted')
        # plt.tight_layout()
        plt.title('Mean episodic return')
        plt.ylabel('')
        plt.ylim(-1000, 20)
        plt.xlabel('training iterations')
        plt.tight_layout()
        # plt.show()
        figpath = '/home/mht/PycharmProjects/rllib_random_feature/fig/' + y + '.png'
        plt.savefig(figpath)

def plot_pendulum():
    sns.set(style='darkgrid', font_scale=1.3)
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
