import os
import numpy as np
import h5py
import scipy.misc

import config

def createH5Test(params):

    # create output h5 file
    output_h5 = '%s_%d_test.h5' %(params['name'], params['img_resize'])
    f_h5 = h5py.File(output_h5, "w")

    # read data info from lists
    list_im = []
    list_lab = []

    for x in range(300):
        path = 'test/' + str(1).zfill(8)[-8:] + '.jpg'
        list_im.append(os.path.join(params['data_root'], path))

    list_im = np.array(list_im, np.object)
    N = list_im.shape[0]
    print('# Images found:', N)
    
    im_set = f_h5.create_dataset("images", (N,params['img_resize'],params['img_resize'],3), dtype='uint8') # space for resized images

    for i in range(N):
        image = scipy.misc.imread(list_im[i])
        assert image.shape[2]==3, 'Channel size error!'
        image = scipy.misc.imresize(image, (params['img_resize'],params['img_resize']))

        im_set[i] = image

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))

    f_h5.close()

if __name__=='__main__':
    params_test = {
        'name': 'miniplaces',
        'split': 'test',
        'img_resize': config.size,
        'data_root': config.data_root,  # MODIFY PATH ACCORDINGLY
            'data_list': config.data_val_list
    }

    createH5Test(params_test)