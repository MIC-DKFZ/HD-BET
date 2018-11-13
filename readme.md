#HD-BET

This repository provides easy to use access to our recently published HD-BET brain extraction tool. HD-BET is he result
of a joint project between the Department of Neuroradiology of the Heidelberg University Medical Center and the 
Division of Medical Image Computing of the German Cancer Research Center.

ADVANTAGES OF HD-BET
 
##Installation Instructions

1) Clone this repository
2) Go into the HD_BET directory and install with
    ```
    pip install -e .
    ```
3) Download the model parameters from 
https://www.dropbox.com/s/xsm95642suvq7n0/HD-BET_parameters.zip?dl=0
4) Extract the model parameters to HD_BET/parameters/0.model etc.  
If you wish to save them in a different folder you can do so. In that case please modify the path to the parameters in 
HD_BET/paths.py and make sure you installed HD_BET with the '-e' option.


## How to use it
Using HD_BET is quite straightforward. We provide CPU as well as GPU support. Running on GPU is a lot faster though 
and should always be preferred. Here is a minimalistic example of how you can use HD-BET (you need to be in the HD_BET 
directory)

```bash
python run.py INPUT_FILENAME OUTPUT_FILENAME
```
INPUT_FILENAME must be a nifti (.nii.gz) file containing 3D image data. 4D image sequences are not supported.

By default, this will run in GPU mode, use the parameters of fold 0 ('0.model'), use test time data augmentation by 
mirroring along all axes and not do any postprocessing.

For batch processing it is faster to process an entire folder at once as this will mitigate the overhead of loading 
and initializing the model for each case:

```bash
python run.py INPUT_FOLDER OUTPUT_FOLDER
```

This command will look for all nifti files (*.nii.gz) in the INPUT_FOLDER and save the brain masks under the same name
in OUTPUT_FOLDER.

**We recommend to run all brain extractions with this command to make use of an ensemble of five models. This will 
still run very quickly on GPU**
```bash
python run.py INPUT_FOLDER OUTPUT_FOLDER -mode accurate
```


To modify the device (CPU/GPU), whether to use test time data augmentation and postprocessing please refer 
to the documentation of run.py:

```bash
python run.py --help
```

## FAQ
1) **How much GPU memory do I need to run HD-BET?**  
We ran all our experiments on NVIDIA Titan X GPUs with 12 GB memory. For inference you will need less, but since 
inference in implemented by exploiting the fully convolutional nature of CNNs the amount of memory required depends on 
your image. Typical image sshould run with less than 4 GB of GPU memory consumption. If you run into out of memory
problems please check the following: 1) Make sure the voxel spacing of your data is correct and 2) Ensure your MRI 
image only contains the head region
2) **Will you provide the training code as well?**  
No. The training code is tightly wound around the data which we cannot make public.
3) **What run time can I expect on CPU/GPU?**  
This depends on your MRI image size. Typical run times (preprocessing, postprocessing and resampling included) are just
 a couple of seconds for GPU and about 2 Minutes on CPU (using ```-tta 0 -mode fast```)