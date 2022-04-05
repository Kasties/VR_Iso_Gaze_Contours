import numpy as np
import matplotlib.pyplot as plt
import argparse

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import matplotlib.image as mpimg
import os
from pathlib import Path

mse_distribution_gaussian_list=[]
ssim_distribution_gaussian_list=[]
mse_distribution_autoencoder_list=[]
ssim_distribution_autoencoder_list=[]
mse_distribution_afm_list=[]
ssim_distribution_afm_list=[]


def compare(path, plot):

    distribution_data_path = os.path.join(path, 'distributions') 
    gauusian_data_path = os.path.join(path, 'gaussians') 
    prediction_data_path = os.path.join(path, 'predictions') 
    afm_data_path = os.path.join(path, 'afm')
    img_afm = mpimg.imread(os.path.join(afm_data_path,'afm_predicted.png'))

    files =[ i for i in  sorted(os.listdir(distribution_data_path)) if 'csv' not in i]

    for index,file in enumerate(files):
        img_distribution = mpimg.imread(os.path.join(distribution_data_path,file))
        img_gaussian = mpimg.imread(os.path.join(gauusian_data_path,file))
        img_autoencoder = mpimg.imread(os.path.join(prediction_data_path,file))

        # img_autoencoder = img_autoencoder*img_autoencoder*img_autoencoder
        # img_autoencoder = img_autoencoder/img_autoencoder.max()

        # img_gaussian = img_gaussian*img_gaussian*img_gaussian
        # img_gaussian = img_gaussian/img_gaussian.max()

        rows, cols = img_distribution.shape

        mse_none = mean_squared_error(img_distribution, img_distribution)
        ssim_none = ssim(img_distribution, img_distribution, data_range=img_distribution.max() - img_distribution.min())

        mse_distribution_gaussian = mean_squared_error(img_distribution, img_gaussian)
        ssim_distribution_gaussian = ssim(img_distribution, img_gaussian,
                        data_range=img_gaussian.max() - img_gaussian.min())

        mse_distribution_autoencoder = mean_squared_error(img_distribution, img_autoencoder)
        ssim_distribution_autoencoder = ssim(img_distribution, img_autoencoder,
                        data_range=img_autoencoder.max() - img_autoencoder.min())
     
        mse_distribution_afm = mean_squared_error(img_distribution, img_afm)
        ssim_distribution_afm = ssim(img_distribution, img_afm,
                        data_range=img_afm.max() - img_afm.min())

        if(False):

            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4),
                                    sharex=True, sharey=True)
            ax = axes.ravel()


            ax[0].imshow(img_distribution, cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[0].set_xlabel(f'MSE: {mse_none:.5f}, SSIM: {ssim_none:.5f}')
            ax[0].set_title('Original image')

            ax[1].imshow(img_gaussian, cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[1].set_xlabel(f'MSE: {mse_distribution_gaussian:.5f}, SSIM: {ssim_distribution_gaussian:.5f}')
            ax[1].set_title('Gaussian')

            ax[2].imshow(img_autoencoder, cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[2].set_xlabel(f'MSE: {mse_distribution_autoencoder:.5f}, SSIM: {ssim_distribution_autoencoder:.5f}')
            ax[2].set_title('Autoencoder')

            ax[3].imshow(img_afm, cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[3].set_xlabel(f'MSE: {mse_distribution_afm:.5f}, SSIM: {ssim_distribution_afm:.5f}')
            ax[3].set_title('AFM')

            plt.tight_layout()
            plt.show()
        
   

        mse_distribution_gaussian_list.append(mse_distribution_gaussian)
        ssim_distribution_gaussian_list.append(ssim_distribution_gaussian)
        mse_distribution_autoencoder_list.append(mse_distribution_autoencoder)
        ssim_distribution_autoencoder_list.append(ssim_distribution_autoencoder)
        mse_distribution_afm_list.append(mse_distribution_afm)
        ssim_distribution_afm_list.append(ssim_distribution_afm)

def main(args):
 
    
    compare(args.path, args.plot)

    ssim_distribution_gaussian_list_notnan =  np.array(ssim_distribution_gaussian_list)
    ssim_distribution_gaussian_list_notnan = ssim_distribution_gaussian_list_notnan[~ np.isnan(ssim_distribution_gaussian_list_notnan)]

    string = f' gaussian MSE: {np.mean(mse_distribution_gaussian_list):.5f}, gaussian SSIM: {np.mean(ssim_distribution_gaussian_list_notnan):.5f} \n'
    string += f' autoencoder MSE: {np.mean(mse_distribution_autoencoder_list):.5f}, autoencoder SSIM: {np.mean(ssim_distribution_autoencoder_list):.5f}\n'
    string += f' afm MSE: {np.mean(mse_distribution_afm_list):.5f}, afm SSIM: {np.mean(ssim_distribution_afm_list):.5f}'

    print(string)

    mse_gaussian_winning = np.array(mse_distribution_gaussian_list)[np.array(mse_distribution_gaussian_list)<np.array(mse_distribution_autoencoder_list)]
    mse_preddiction_loosing = np.array(mse_distribution_autoencoder_list)[np.array(mse_distribution_gaussian_list)<np.array(mse_distribution_autoencoder_list)]
    SSIM_gaussian_winning = np.array(ssim_distribution_gaussian_list)[np.array(ssim_distribution_gaussian_list)>np.array(ssim_distribution_autoencoder_list)]

    print(mse_gaussian_winning,mse_preddiction_loosing)
    print(SSIM_gaussian_winning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--plot", type=str, required=False, default=False)
    args = parser.parse_args()
    main(args)

