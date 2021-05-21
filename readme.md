# HD-BET 

This repository provides easy to use access to our recently published HD-BET brain extraction tool. HD-BET is the result
of a joint project between the Department of Neuroradiology at the Heidelberg University Hospital and the 
Division of Medical Image Computing at the German Cancer Research Center (DKFZ).

If you are using HD-BET, please cite the following publication: 

Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W, 
Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial neural 
networks. Hum Brain Mapp. 2019; 1–13. https://doi.org/10.1002/hbm.24750

Compared to other commonly used brain extraction tools, HD-BET has some significant advantages:
- HD-BET was developed with MRI-data from a large multicentric clinical trial in adult brain tumor patients acquired 
across 37 institutions in Europe and included a broad range of MR hardware and acquisition parameters, pathologies 
or treatment-induced tissue alterations. We used 2/3 of data for training and validation and 1/3 for testing. 
Moreover independent testing of HD-BET was performed in three public benchmark datasets (NFBS, LPBA40 and CC-359).
- HD-BET was trained with precontrast T1-w, postcontrast T1-w, T2-w and FLAIR sequences. It can perform independent 
brain extraction on various different MRI sequences and is not restricted to precontrast T1-weighted (T1-w) sequences.
 Other MRI sequences may work as well (just give it a try!)
- HD-BET was designed to be robust with respect to brain tumors, lesions and resection cavities as well as different 
MRI scanner hardware and acquisition parameters.
- HD-BET outperformed five publicly available brain extraction algorithms (FSL BET, AFNI 3DSkullStrip, Brainsuite BSE, 
ROBEX and BEaST) across all datasets and yielded median improvements of +1.33 to +2.63 points for the DICE 
coefficient and -0.80 to -2.75 mm for the Hausdorff distance (Bonferroni-adjusted p<0.001). 
- HD-BET is very fast on GPU with <10s run time per MRI sequence. Even on CPU it is not slower than other commonly 
used tools.

## Installation Instructions 
Note that you need to have a python3 installation for HD-BET to work. Please also make sure to install HD-BET with the
correct pip version (the one that is connected to python3). You can verify this using the `--version` command:

```
(dl_venv) fabian@Fabian:~$ pip --version
pip 20.0.2 from /home/fabian/dl_venv/lib/python3.6/site-packages/pip (python 3.6)
```

If it does not show python 3.X, you can try pip3. If that also does not work you probably need to install python3 first.

Once python 3 and pip are set up correctly, run the following commands to install HD-BET:
1) Clone this repository:
    ```bash
    git clone https://github.com/MIC-DKFZ/HD-BET
    ```
2) Go into the repository (the folder with the setup.py file) and install:
    ```
    cd HD-BET
    pip install -e .
    ```
3) Per default, model parameters will be downloaded to ~/hd-bet_params. If you wish to use a different folder, open 
HD_BET/paths.py in a text editor and modify ```folder_with_parameter_files```


## How to use it 

Using HD_BET is straightforward. You can use it in any terminal on your linux system. The hd-bet command was installed 
automatically. We provide CPU as well as GPU support. Running on GPU is a lot faster though 
and should always be preferred. Here is a minimalistic example of how you can use HD-BET (you need to be in the HD_BET 
directory)

```bash
hd-bet -i INPUT_FILENAME
```

INPUT_FILENAME must be a nifti (.nii.gz) file containing 3D MRI image data. 4D image sequences are not supported 
(however can be splitted upfront into the individual temporal volumes using fslsplit<sup>1</sup>). 
INPUT_FILENAME can be either a pre- or postcontrast T1-w, T2-w or FLAIR MRI sequence. Other modalities might work as well.
Input images must match the orientation of standard MNI152 template! Use fslreorient2std <sup>2</sup> upfront to ensure 
that this is the case.

By default, HD-BET will run in GPU mode, use the parameters of all five models (which originate from a five-fold 
cross-validation), use test time data augmentation by mirroring along all axes and not do any postprocessing.

For batch processing it is faster to process an entire folder at once as this will mitigate the overhead of loading 
and initializing the model for each case:

```bash
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER
```

The above command will look for all nifti files (*.nii.gz) in the INPUT_FOLDER and save the brain masks under the same name
in OUTPUT_FOLDER.

### GPU is nice, but I don't have one of those... What now? 

HD-BET has CPU support. Running on CPU takes a lot longer though and you will need quite a bit of RAM. To run on CPU, 
we recommend you use the following command:

```bash
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER -device cpu -mode fast -tta 0
```
This works of course also with just an input file:

```bash
hd-bet -i INPUT_FILENAME -device cpu -mode fast -tta 0
```

The options *-mode fast* and *-tta 0* will disable test time data augmentation (speedup of 8x) and use only one model instead of an ensemble of five models 
for the prediction.

### More options:
For more information, please refer to the help functionality:

```bash
hd-bet --help
```

## FAQ

1) **How much GPU memory do I need to run HD-BET?**  
We ran all our experiments on NVIDIA Titan X GPUs with 12 GB memory. For inference you will need less, but since 
inference in implemented by exploiting the fully convolutional nature of CNNs the amount of memory required depends on 
your image. Typical image should run with less than 4 GB of GPU memory consumption. If you run into out of memory
problems please check the following: 1) Make sure the voxel spacing of your data is correct and 2) Ensure your MRI 
image only contains the head region
2) **Will you provide the training code as well?**  
No. The training code is tightly wound around the data which we cannot make public.
3) **What run time can I expect on CPU/GPU?**  
This depends on your MRI image size. Typical run times (preprocessing, postprocessing and resampling included) are just
 a couple of seconds for GPU and about 2 Minutes on CPU (using ```-tta 0 -mode fast```)


<sup>1</sup>https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils

<sup>2</sup>https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained