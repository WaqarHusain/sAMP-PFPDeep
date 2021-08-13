import math
from os import makedirs
from os import walk
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras_preprocessing import image
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from PIL import Image


FASTA_INPUT_FILE_NAME = "./Example.fasta"


encoder = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13,
           'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0}
frequency = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0,
             'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, 'X': 0}
hydropathy = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
              'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9,
              'Y': -1.3, 'X': 0.0}
volume = {'A': 91.5, 'C': 118.0, 'D': 124.5, 'E': 155.1, 'F': 203.4, 'G': 66.4, 'H': 167.3, 'I': 168.8, 'K': 171.3,
          'L': 167.9, 'M': 170.8, 'N': 135.2, 'P': 129.3, 'Q': 161.1, 'R': 202.0, 'S': 99.1, 'T': 122.1, 'V': 141.7,
          'W': 237.6, 'Y': 203.6, 'X': 0.0}
polarity = {'A': 0.0, 'C': 1.48, 'D': 40.7, 'E': 49.91, 'F': 0.35, 'G': 0.0, 'H': 51.6, 'I': 0.15, 'K': 49.5, 'L': 0.45,
            'M': 1.43, 'N': 3.38, 'P': 1.58, 'Q': 3.53, 'R': 52.0, 'S': 1.67, 'T': 1.66, 'V': 0.13, 'W': 2.1, 'Y': 1.61,
            'X': 0.0}
pK_side_chain = {'A': 0.0, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.0, 'G': 0.0, 'H': 6.0, 'I': 0.0, 'K': 10.53,
                 'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 12.48, 'S': 0.0, 'T': 0.0, 'V': 0.0, 'W': 0.0,
                 'Y': 10.7, 'X': 0.0}
prct_exposed_residues = {'A': 15.0, 'C': 5.0, 'D': 50.0, 'E': 55.0, 'F': 10.0, 'G': 10.0, 'H': 34.0, 'I': 13.0,
                         'K': 85.0, 'L': 16.0, 'M': 20.0, 'N': 49.0, 'P': 45.0, 'Q': 56.0, 'R': 67.0, 'S': 32.0,
                         'T': 32.0, 'V': 14.0, 'W': 17.0, 'Y': 41.0, 'X': 0.0}
hydrophilicity = {'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5, 'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0,
                  'L': -1.8, 'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0, 'S': 0.3, 'T': -0.4, 'V': -1.5,
                  'W': -3.4, 'Y': -2.3, 'X': 0.0}
accessible_surface_area = {'A': 27.8, 'C': 15.5, 'D': 60.6, 'E': 68.2, 'F': 25.5, 'G': 24.5, 'H': 50.7, 'I': 22.8,
                           'K': 103.0, 'L': 27.6, 'M': 33.5, 'N': 60.1, 'P': 51.5, 'Q': 68.7, 'R': 94.7, 'S': 42.0,
                           'T': 45.0, 'V': 23.7, 'W': 34.7, 'Y': 55.2, 'X': 0.0}
refractivity = {'A': 4.34, 'C': 35.77, 'D': 12.0, 'E': 17.26, 'F': 29.4, 'G': 0.0, 'H': 21.81, 'I': 19.06, 'K': 21.29,
                'L': 18.78, 'M': 21.64, 'N': 13.28, 'P': 10.93, 'Q': 17.56, 'R': 26.66, 'S': 6.35, 'T': 11.01,
                'V': 13.92, 'W': 42.53, 'Y': 31.53, 'X': 0.0}
local_flexibility = {'A': 705.42, 'C': 2412.5601, 'D': 34.96, 'E': 1158.66, 'F': 5203.8599, 'G': 33.18, 'H': 1637.13,
                     'I': 5979.3701, 'K': 699.69, 'L': 4985.73, 'M': 4491.6602, 'N': 513.4601, 'P': 431.96,
                     'Q': 1087.83, 'R': 1484.28, 'S': 174.76, 'T': 601.88, 'V': 4474.4199, 'W': 6374.0698,
                     'Y': 4291.1001, 'X': 0.0}
accessible_surface_area_folded = {'A': 31.5, 'C': 13.9, 'D': 60.9, 'E': 72.3, 'F': 28.7, 'G': 25.2, 'H': 46.7,
                                  'I': 23.0, 'K': 110.3, 'L': 29.0, 'M': 30.5, 'N': 62.2, 'P': 53.7, 'Q': 74.0,
                                  'R': 93.8, 'S': 44.2, 'T': 46.0, 'V': 23.5, 'W': 41.7, 'Y': 59.1, 'X': 0.0}
mass = {'A': 70.079, 'C': 103.144, 'D': 115.089, 'E': 129.116, 'F': 147.177, 'G': 57.052, 'H': 137.142, 'I': 113.16,
        'K': 128.174, 'L': 113.16, 'M': 131.198, 'N': 114.104, 'P': 97.177, 'Q': 128.131, 'R': 156.188, 'S': 87.078,
        'T': 101.105, 'V': 99.133, 'W': 186.213, 'Y': 163.17, 'X': 0.0}
solvent_exposed_area = {'A': 0.48, 'C': 0.32, 'D': 0.81, 'E': 0.93, 'F': 0.42, 'G': 0.51, 'H': 0.66, 'I': 0.39,
                        'K': 0.93, 'L': 0.41, 'M': 0.44, 'N': 0.82, 'P': 0.78, 'Q': 0.81, 'R': 0.84, 'S': 0.7,
                        'T': 0.71, 'V': 0.4, 'W': 0.49, 'Y': 0.67, 'X': 0.0}


def seqToMat(seq):
    n = dim
    seq_Mat = [[0 for x in range(n)] for y in range(n)]
    hydropathy_Mat = [[0 for x in range(n)] for y in range(n)]
    volume_Mat = [[0 for x in range(n)] for y in range(n)]
    polarity_Mat = [[0 for x in range(n)] for y in range(n)]
    pK_side_chain_Mat = [[0 for x in range(n)] for y in range(n)]
    prct_exposed_residues_Mat = [[0 for x in range(n)] for y in range(n)]
    hydrophilicity_Mat = [[0 for x in range(n)] for y in range(n)]
    accessible_surface_area_Mat = [[0 for x in range(n)] for y in range(n)]
    refractivity_Mat = [[0 for x in range(n)] for y in range(n)]
    local_flexibility_Mat = [[0 for x in range(n)] for y in range(n)]
    accessible_surface_area_folded_Mat = [[0 for x in range(n)] for y in range(n)]
    mass_Mat = [[0 for x in range(n)] for y in range(n)]
    solvent_exposed_area_Mat = [[0 for x in range(n)] for y in range(n)]

    seqiter = 0
    for i in range(n):
        for j in range(n):
            if seqiter < len:
                try:
                    seq_Mat_v = encoder[seq[seqiter]]
                    hydropathy_Mat_v = hydropathy[seq[seqiter]]
                    volume_Mat_v = volume[seq[seqiter]]
                    polarity_Mat_v = polarity[seq[seqiter]]
                    pK_side_chain_Mat_v = pK_side_chain[seq[seqiter]]
                    prct_exposed_residues_Mat_v = prct_exposed_residues[seq[seqiter]]
                    hydrophilicity_Mat_v = hydrophilicity[seq[seqiter]]
                    accessible_surface_area_Mat_v = accessible_surface_area[seq[seqiter]]
                    refractivity_Mat_v = refractivity[seq[seqiter]]
                    local_flexibility_Mat_v = local_flexibility[seq[seqiter]]
                    accessible_surface_area_folded_Mat_v = accessible_surface_area_folded[seq[seqiter]]
                    mass_Mat_v = mass[seq[seqiter]]
                    solvent_exposed_area_Mat_v = solvent_exposed_area[seq[seqiter]]
                except ValueError:
                    exit(0)
                else:
                    seq_Mat[i][j] = seq_Mat_v
                    hydropathy_Mat[i][j] = hydropathy_Mat_v
                    volume_Mat[i][j] = volume_Mat_v
                    polarity_Mat[i][j] = polarity_Mat_v
                    pK_side_chain_Mat[i][j] = pK_side_chain_Mat_v
                    prct_exposed_residues_Mat[i][j] = prct_exposed_residues_Mat_v
                    hydrophilicity_Mat[i][j] = hydrophilicity_Mat_v
                    accessible_surface_area_Mat[i][j] = accessible_surface_area_Mat_v
                    refractivity_Mat[i][j] = refractivity_Mat_v
                    local_flexibility_Mat[i][j] = local_flexibility_Mat_v
                    accessible_surface_area_folded_Mat[i][j] = accessible_surface_area_folded_Mat_v
                    mass_Mat[i][j] = mass_Mat_v
                    solvent_exposed_area_Mat[i][j] = solvent_exposed_area_Mat_v
                seqiter += 1
    return np.asarray(seq_Mat), np.asarray(
        np.matrix(hydropathy_Mat) + np.matrix(volume_Mat) + np.matrix(polarity_Mat) + np.matrix(pK_side_chain_Mat) + \
        np.matrix(prct_exposed_residues_Mat) + np.matrix(hydrophilicity_Mat) + np.matrix(accessible_surface_area_Mat) + \
        np.matrix(refractivity_Mat) + np.matrix(local_flexibility_Mat) + np.matrix(accessible_surface_area_folded_Mat) \
        + np.matrix(mass_Mat) + np.matrix(solvent_exposed_area_Mat))


def freqMat(seq):
    n = dim
    freq_Mat = [[0 for x in range(n)] for y in range(n)]
    X = ProteinAnalysis(str(seq))
    frequency['A'] = X.count_amino_acids()['A']
    frequency['C'] = X.count_amino_acids()['C']
    frequency['D'] = X.count_amino_acids()['D']
    frequency['E'] = X.count_amino_acids()['E']
    frequency['F'] = X.count_amino_acids()['F']
    frequency['G'] = X.count_amino_acids()['G']
    frequency['H'] = X.count_amino_acids()['H']
    frequency['I'] = X.count_amino_acids()['I']
    frequency['K'] = X.count_amino_acids()['K']
    frequency['L'] = X.count_amino_acids()['L']
    frequency['M'] = X.count_amino_acids()['M']
    frequency['N'] = X.count_amino_acids()['N']
    frequency['P'] = X.count_amino_acids()['P']
    frequency['Q'] = X.count_amino_acids()['Q']
    frequency['R'] = X.count_amino_acids()['R']
    frequency['S'] = X.count_amino_acids()['S']
    frequency['T'] = X.count_amino_acids()['T']
    frequency['V'] = X.count_amino_acids()['V']
    frequency['W'] = X.count_amino_acids()['W']
    frequency['Y'] = X.count_amino_acids()['Y']

    seqiter = 0
    for i in range(n):
        for j in range(n):
            if seqiter < len:
                try:
                    freq_Mat_v = frequency[seq[seqiter]]
                except ValueError:
                    exit(0)
                else:
                    freq_Mat[i][j] = freq_Mat_v
                seqiter += 1
    return np.asarray(freq_Mat)

def define_model_VGG():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.load_weights('newWeightsVGG.h5')
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

iter = 0
makedirs('./seqImgs/', exist_ok=True)

for seq_record in SeqIO.parse(FASTA_INPUT_FILE_NAME, "fasta"):
    seq = str(seq_record.seq)
    len = seq.__len__()
    dim = int(math.ceil(math.sqrt(len)))
    seq = seq.replace('B', 'D')
    seq = seq.replace('U', 'C')
    seq = seq.replace('Z', 'Q')
    seq = seq.replace('O', 'K')
    r = seqToMat(seq)[0]
    rNew = np.interp(r, (r.min(), r.max()), (0, 256))
    g = seqToMat(seq)[1]
    gNew = np.interp(g, (g.min(), g.max()), (0, 256))
    b = freqMat(seq)
    bNew = np.interp(b, (b.min(), b.max()), (0, 256))
    rgbArray = np.zeros((dim, dim, 3), 'uint8')
    rgbArray[..., 0] = rNew
    rgbArray[..., 1] = gNew
    rgbArray[..., 2] = bNew
    img = Image.fromarray(rgbArray)
    img.save("./seqImgs/" + str(iter) + '.jpeg')
    iter += 1

model = define_model_VGG()
data_dir = './seqImgs/'
_, dirs, filenames = next(walk(data_dir))
my_list = []

for f in filenames:
    img = np.expand_dims(image.img_to_array(image.load_img(data_dir+f, target_size=(200, 200))), axis=0)
    y_pred = model.predict(img)
    my_list.append(round(y_pred[0][0]))

with open('Predicted Labels.csv', 'w') as pred:
    for item in my_list:
        pred.write("%s\n" % item)

