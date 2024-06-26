import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd


sns.set_theme(style="whitegrid")
n_radi = 8
radi_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
# simulation runs for each radius
n_ins = 10
# steps
n_step = 100
# number of radius


KL_poi_diff = np.zeros((n_radi, n_ins, n_step))
KL_vel_diff = np.zeros((n_radi, n_ins, n_step))

for r_idx in range(n_radi):
    radius = radi_arr[r_idx]
    KL_dir = "./logs/KL_track_radi_{}".format(radius)
    with open('{}/obs.pickle'.format(KL_dir), 'rb') as fp:
        obs = pickle.load(fp)

    with open('{}/unobs_state.pickle'.format(KL_dir), 'rb') as fp:
        unobs = pickle.load(fp)

    with open('{}/est_state.pickle'.format(KL_dir), 'rb') as fp:
        KL_est = pickle.load(fp)

    with open('{}/EM_est.pickle'.format(KL_dir), 'rb') as fp:
        EM_est = pickle.load(fp)


    # with open('{}/obs.pickle'.format(ot_dir), 'rb') as fp:
    #     ot_obs = pickle.load(fp)
    #
    # with open('{}/unobs_state.pickle'.format(ot_dir), 'rb') as fp:
    #     ot_unobs = pickle.load(fp)
    #
    # with open('{}/est_state.pickle'.format(ot_dir), 'rb') as fp:
    #     ot_est = pickle.load(fp)
    #
    # with open('{}/EM_est.pickle'.format(ot_dir), 'rb') as fp:
    #     ot_EM_est = pickle.load(fp)

    obs = np.array(obs)
    unobs = np.array(unobs)
    EM_est = np.array(EM_est)
    KL_est = np.array(KL_est)


    # ot_obs, ot_unobs are the same as these of COT
    KL_poi_err = np.sqrt(np.sum((KL_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))
    em_poi_err = np.sqrt(np.sum((EM_est[:, :, :2] - unobs[:, :, :2]) ** 2, axis=2))

    KL_vel_err = np.sqrt(np.sum((KL_est[:, :, 2:] - unobs[:, :, 2:]) ** 2, axis=2))
    em_vel_err = np.sqrt(np.sum((EM_est[:, :, 2:] - unobs[:, :, 2:]) ** 2, axis=2))

    KL_poi_diff[r_idx, :, :] = KL_poi_err - em_poi_err
    KL_vel_diff[r_idx, :, :] = KL_vel_err - em_vel_err

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.kdeplot(cot_poi_diff.reshape(-1), label='COT - EM')
# sns.kdeplot(ot_poi_diff.reshape(-1), label='OT - EM')
# ax.set_xlim(-8, 5)
# # # plt.xlabel('Steps', fontsize=16)
# # # plt.ylabel('Velocity errors difference', fontsize=16)
# # # plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
# plt.legend(bbox_to_anchor=(1, 1), title='', fontsize=16, title_fontsize=16)
# # # plt.savefig('vel_diff.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
# plt.show()


############# Position error ##############
# errdf = []
# total_n = n_radi*n_ins*n_step
# errdf.append(pd.DataFrame({'Algorithm':['COT - EM',]*total_n, 'Difference': cot_poi_diff.reshape(-1)}))
# errdf.append(pd.DataFrame({'Algorithm':['OT - EM',]*total_n, 'Difference': ot_poi_diff.reshape(-1)}))
# errdf = pd.concat(errdf)
# errdf = errdf.reset_index(drop=True)
#
# plt.subplots(figsize=(8, 6))
# ax = sns.ecdfplot(data=errdf, x='Difference', hue='Algorithm')
# ax.set_xlim(-10, 6)
# plt.xlabel('Difference', fontsize=14)
# plt.ylabel('Proportion', fontsize=14)
# plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
# plt.rcParams['title_fontsize'] = 16
# plt.rcParams['fontsize'] = 16
# ax.legend(bbox_to_anchor=(0.1, 1), title='Algorithm', fontsize=16, title_fontsize=16)
# plt.show()
# plt.savefig('poi_diff.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
KLPoi_mean = np.mean(KL_poi_diff, axis=(1, 2))
KLPoi_std = np.std(KL_poi_diff, axis=(1, 2))

print('Mean KL Posi difference', np.round(KLPoi_mean, 4))
print('STD KL Posi difference', np.round(KLPoi_std, 4))
print('KL Mean/STD ratio', np.round(np.divide(KLPoi_mean, KLPoi_std), 4))

# print('Mean OT Posi difference', np.round(np.mean(ot_poi_diff, axis=(1, 2)), 4))
# print('Var OT Posi difference', np.round(np.var(ot_poi_diff, axis=(1, 2)), 4))

################ Velocity error #################
# errdf1 = []
# errdf1.append(pd.DataFrame({'Algorithm':['COT - EM',]*total_n, 'Difference': cot_vel_diff.reshape(-1)}))
# errdf1.append(pd.DataFrame({'Algorithm':['OT - EM',]*total_n, 'Difference': ot_vel_diff.reshape(-1)}))
# errdf1 = pd.concat(errdf1)
# errdf1 = errdf1.reset_index(drop=True)
#
# plt.subplots(figsize=(8, 6))
# ax = sns.ecdfplot(data=errdf1, x='Difference', hue='Algorithm')
# ax.set_xlim(-10, 6)
# plt.xlabel('Difference', fontsize=14)
# plt.ylabel('Proportion', fontsize=14)
# plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
# plt.rcParams['title_fontsize'] = 16
# plt.rcParams['fontsize'] = 16
# ax.legend(bbox_to_anchor=(0.1, 1), title='Algorithm', fontsize=16, title_fontsize=16)
# plt.show()
# plt.savefig('vel_diff.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)


KLVel_mean = np.mean(KL_vel_diff, axis=(1, 2))
KLVel_std = np.std(KL_vel_diff, axis=(1, 2))

print('Mean KL Vel difference', np.round(KLVel_mean, 4))
print('STD KL Vel difference', np.round(KLVel_std, 4))
print('KL Mean/STD ratio', np.round(np.divide(KLVel_mean, KLVel_std), 4))
# print('Mean OT Vel difference', np.round(np.mean(ot_vel_diff, axis=(1, 2)), 4))
# print('Var OT Vel difference', np.round(np.var(ot_vel_diff, axis=(1, 2)), 4))
