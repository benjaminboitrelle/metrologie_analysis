# -*- coding: utf-8 -*-

import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

import scipy.io as spio
from lmfit.models import GaussianModel, ConstantModel


class MetrologieAnalyse(object):

    def __init__(self, input_fname, input_no_slit, output_dir, energy):
        self._input_fname = input_fname
        self._input_no_slit = input_no_slit
        self._output_dir = output_dir
        self._energy = energy

    def _get_input_fname(self, input_fname_templ, energy):
        ''' File template: {slit_postion}_I{illumination}_{energy}eV.mat
        '''
        input_fname = input_fname_templ.format(slits="*",
                                               illumination="*",
                                               energy="*")

        # List all files contained in the folder
        files = glob.glob(input_fname)

        # Extract a prefix and a suffix according to keyword energy
        prefix, suffix = input_fname_templ.split("{energy}")
        prefix = prefix.format(slits="SlitD",
                               illumination="dark")

        prefix_pos, suffix_pos = input_fname_templ.split("{slits}")
        start_pos = len(prefix_pos)
        end_pos = len(prefix_pos) + 5

        energies = [int(f[len(prefix):-len(suffix)]) for f in files]

        searched_files = {energy: [f for f in files
                                   if int(f[len(prefix):
                                            -len(suffix)]) == energy]
                          for energy in energies}

        # Reorder dictionary into 2 layers: {energy: {slit_position: files}}
        dict_files = {}
        slits = ["SlitG", "SlitD"]
        for key, value in searched_files.items():
            dict_files[key] = {}
            for slit in slits:
                data = []
                for v in value:
                    if v[start_pos:end_pos] == slit:
                        dict_files[key][slit] = {}
                        if v.find('beam') != -1:
                            data.append((v, 'img'))
                        if v.find('dark') != -1:
                            data.append((v, 'imgDark'))
                    dict_files[key][slit] = data

        return dict_files

    def load_images(self, input_fname, item):
        ''' For a given exposure time return a stack of arrays corresponding
            to the 2 images taken
        '''

        img_matlab = spio.loadmat(input_fname, squeeze_me=False)
        img = img_matlab[item]

        return img

    def average_image(self, img):
        ''' Average the input image, having 3 dimensions
            (n_rows, n_cols, n_frames) according to the thirds axis (frame)
        '''

        return np.mean(img, axis=2)

    def crop_image(self, data, roi_size):
        ''' For a given image, search the maximum value  coordinates and
            define a determine roi around this maximum
        '''

        max_d = np.amax(data)
        position_max = np.where(data == max_d)
        roi_u = (int(position_max[0] - roi_size),
                 int(position_max[0] + roi_size))
        roi_v = (int(position_max[1] - roi_size),
                 int(position_max[1] + roi_size))

        return data[roi_u[0]:roi_u[1], roi_v[0]:roi_v[1]]

    def get_lsf(self, data):
        v_slice = 115
        return np.diff(data[v_slice])

    def gaussian(self, x, amp, cen, wid):
        return amp * np.exp(-(x-cen)**2 / wid)

    def get_data(self, files, slit_position, img_type):
        ''' Calculate mean of an image for one energy, a slit position

            Parameters:
            ----------
                files: path to images to open
                slit_position: 'slitD' or 'slitG'
                img_type: 'img' or 'imgDark'

            Return:
            ------
                Dictionary
        '''

        dict_mean = {}
        for key, value in files.items():
            dict_mean[key] = {}
            for subkey, subvalue in value.items():
                if subkey == slit_position:
                    for files in subvalue:
                        if files[1] == img_type:
                            img_mean = np.mean(self.load_images(files[0],
                                                                files[1]),
                                               axis=2)
                    dict_mean[key] = img_mean

        return dict_mean

    def run(self):

        files = self._get_input_fname(self._input_fname, self._energy)
#        self.get_data(files, 'SlitG')
        file_no_slit = self._get_input_fname(self._input_no_slit, self._energy)

        dat_l = {}
        for key, value in files.items():
            dat_l[key] = {}
            for subkey, subvalue in value.items():
                if subkey == 'SlitG':
                    for files in subvalue:
                        if files[1] == "img":
                            img_light_l = np.mean(self.load_images(files[0],
                                                                   files[1]),
                                                  axis=2)
                        else:
                            img_dark_l = np.mean(self.load_images(files[0],
                                                                  files[1]),
                                                 axis=2)

                    img_l = img_light_l - img_dark_l
#                    img_l = self.crop_image(img_l, 80)
                    dat_l[key] = img_l

        dat_s = {}
        for key, value in file_no_slit.items():
            dat_s[key] = {}
            for subkey, subvalue in value.items():
                if subkey == 'SlitG':
                    for files in subvalue:
                        if files[1] == "img":
                            img_light = np.mean(self.load_images(files[0],
                                                                 files[1]),
                                                axis=2)
                        else:
                            img_dark = np.mean(self.load_images(files[0],
                                                                files[1]),
                                               axis=2)

                    img = img_light
                    dat_s[key] = img

        fwhm = []
        fwhm_err = []
        energy = []
        for key_slit in (dat_l):
            img_weight = dat_l[key_slit]/dat_s[key_slit]
            img_w = self.crop_image(img_weight, 50)
            plt.imshow(img_w)
            plt.savefig("roi_{}_eV.png".format(key_slit))
            plt.xlabel("Horizontal position [pixel]")
            plt.ylabel("Vertical position [pixel]")
            plt.show()
#            print(img_w.shape)

            lsf = np.diff(img_w[55])
            amplitude = np.max(lsf)
            peak = np.where(lsf == amplitude)
            peak = peak[0] * 11
            x_um = [x*11 for x in np.arange(len(lsf))]

            plt.plot(x_um, lsf)

            model = GaussianModel(prefix='peak_') + ConstantModel()
            params = model.make_params(c=1.0, peak_center=peak,
                                       peak_sigma=22, peak_amplitude=amplitude)
            result = model.fit(lsf, params, x=x_um)
            fwhm.append(2 * np.sqrt(2 * np.log(2)) * result.best_values['peak_sigma'])
            for key in result.params:
                if key == 'peak_fwhm':
                    fwhm_err.append(result.params[key].stderr)

            plt.plot(x_um, result.best_fit)
#            plt.xlim(0, 70)
            plt.xlabel("Horizontal position [µm]")
            plt.ylabel("Counts")
            plt.savefig("fit_{}_ev.png".format(key_slit))
            plt.show()
            energy.append(key_slit)

        plt.errorbar(energy, fwhm, yerr=fwhm_err, fmt='.')
        print("{} +/- {}".format(fwhm, result.params[key].stderr))

        plt.ylim((0, 5))
        plt.xlabel("Energy [eV]")
        plt.ylabel("FWHM [pixel]")
        plt.savefig("fwhm_vs_energy.png")
        plt.show()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        '-i',
                        dest='input_dir_path',
                        type=str,
                        help='Path to the input directory')

    parser.add_argument('--output',
                        '-o',
                        dest='output_dir_path',
                        type=str,
                        help='Path to the output directory')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_arguments()

    output_dir = args.output_dir_path

    input_dir = "/Volumes/LACIE_SHARE/DhyanaMars2019/2019-3-14/scan_11-5-30"
    dir_no_slit = "/Volumes/LACIE_SHARE/DhyanaMars2019/2019-3-14/scanSSBord_13-35-29"
    input_file = "{slits}_I{illumination}_{energy}eV.mat"
    in_fname = os.path.join(input_dir, input_file)
    in_no_slit = os.path.join(dir_no_slit, input_file)
    out_fname = "test"
    energy = 1050

    analyse = MetrologieAnalyse(in_fname, in_no_slit, out_fname, energy)
    analyse.run()
