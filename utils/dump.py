"""
# test()
raw_data_file = mat73.loadmat('../datasets/tvsumm/matlab/ydata-tvsum50.mat')
# print(raw_data_file['tvsum50'].keys())

processed_data = h5py.File('../datasets/tvsumm/helpers/eccv16_dataset_tvsum_google_pool5.h5')


temp1 = {}
frame_data = raw_data_file['tvsum50']['nframes']
for i in range(50):
	val = int(frame_data[i])
	temp1[str(val)] = raw_data_file['tvsum50']['video'][i]

bad_names = list(processed_data.keys())

temp2 = {}
for bn in bad_names:
	key = int(processed_data[bn]['n_frames'][()])
	temp2[str(key)] = bn

# print(temp1)
# print(temp2)
temp3 = {}
for key2 in temp2.keys():
	for key1 in temp1.keys():
		if key1==key2:
			temp3[temp1[key1]] = temp2[key2]

# print(temp3)
print(len(temp3))
# torch.save(temp3, "../datasets/tvsumm/helpers/rename_helper.pt")

import sys
import os
cwd = os.getcwd()
module2add = '\\'.join(cwd.split("\\")[:-1])
sys.path.append(module2add)

from configs.global_configs import cfg as global_config
"""