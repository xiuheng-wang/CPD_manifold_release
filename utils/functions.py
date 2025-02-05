import numpy as np
from scipy.stats import wishart, matrix_normal
import glob
from scipy.io import wavfile, loadmat
import scipy.signal as sig
from sklearn import mixture
import sphfile

def generate_random_SPD_mtx(temp, eigsv):
    """ A function to generate a SPD matrix with given eigenvectors and eigenvalues.
    Usage: matrix = generate_random_SPD_mtx(temp, eigsv)
    Inputs:
    * temp: a matrix to generate eigenvectors
    * eigsv: a vecter with positive eigenvalues
    Outputs:
    * matrix: a SPD matrix."""
    
    temp = np.linalg.svd(temp)[0]
    eigsv = eigsv / np.sum(eigsv) + 1e-6 # positive definite
    matrix = temp @ np.diag(eigsv) @ temp.T
    return matrix

def generate_random_SPD_Wishart(df, scale_matrix):
    """ A function to generate a random SPD matrix from a Wischart distribution.
    Usage: matrix = generate_random_SPD_Wishart(df, scale_matrix)
    Inputs:
        * df: degrees of freedom
        * scale_matrix: a postive definite diagonal matrix
    Outputs:
        * matrix: a random SPD matrix."""
        
    matrix = wishart(df, scale_matrix).rvs()
    return matrix

def generate_random_mtx_normal(M, U, V):
    """ A function to generate a random matrix from a normal distribution.
    Usage: matrix = generate_random_mtx_normal(M, U, V)
    Inputs:
        * M: a matrix
        * U, V: two postive definite matrices
    Outputs:
        * matrix: a random matrix."""
        
    matrix = matrix_normal(M, U, V).rvs()
    return matrix

def import_vad_data(root_path, nb_change, length_noise, length_speech, SNR_convex_coeff, nperseg, sample_factor, no_show):
    fs=16000
    noise_paths = glob.glob(root_path + 'QUT-NOISE\\**\\*.wav', recursive=True)
    speech_paths = glob.glob(root_path + 'TIMIT\\**\\*.WAV', recursive=True)

    X = []
    X_full = []
    index = 0
    while index < nb_change:
        # read speech longer than 3s
        speech_path = speech_paths[int(np.random.rand(1)*len(speech_paths))]
        # print("speech_path:", speech_path)
        sph = sphfile.SPHFile(speech_path)
        sph.open()
        speech_data = sph.content.astype('float64')
        if len(speech_data) < length_speech*fs:
            pass
        else:
            print("Generate No " + str(index) + " time series")
            speech_data = speech_data[:length_speech*fs]
            speech_data *= SNR_convex_coeff/np.max(np.abs(speech_data))
            # noise
            noise_path = noise_paths[int(np.random.uniform()*len(noise_paths))]
            # print("noise_path:", noise_path)
            data = wavfile.read(noise_path)[1]
            rand_start = int(np.random.uniform()*(len(data)-length_noise*fs))
            data = data[rand_start:rand_start+length_noise*fs].astype('float64')
            data *= (1-SNR_convex_coeff)/np.max(np.abs(data))
            data[len(data) - len(speech_data):] += speech_data
            data /= np.max(np.abs(data))
            # stft
            if no_show == 0:
                f, t, Zxx = sig.stft(data, nperseg=nperseg)
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(9, 3))
                plt.pcolormesh(t, f, np.abs(Zxx[1:]), shading='auto')
                print(np.shape(Zxx[1:]))
                plt.show()
            data_stft = np.abs(sig.stft(data, nperseg=nperseg)[2][1:].T)
            # X.append(data_stft[:, ::sample_factor])
            X.append(1.0 / sample_factor * data_stft @ np.kron(np.eye(int(nperseg / (2*sample_factor))), np.ones((sample_factor,1))))
            X_full.append(data_stft)
            index += 1
    return X, X_full