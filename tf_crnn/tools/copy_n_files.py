import shutil
import glob
import os

img_dir = '/home/user/home/cwq/doubao_test_data/croped_text/serialnumber/'
output_dir = '/disk2/cwq/serial_num_train/real'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img_paths = glob.glob(img_dir + '*.jpg')
print('Total: %d' % len(img_paths))
for i in range(200):
    shutil.copy(img_paths[i], output_dir)


