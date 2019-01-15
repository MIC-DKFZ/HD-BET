#HD-BET

This repository provides easy to use access to our recently published HD-BET brain extraction tool. HD-BET is the result
of a joint project between the Department of Neuroradiology at the Heidelberg University Medical Center and the 
Division of Medical Image Computing at the German Cancer Research Center.

Compared to other commonly used brain extraction tools, HD-BET has some significant advantages:
- HD-BET can run brain extraction on the most commom MRI sequences natively and is not restricted to T1w! It was 
trained with T1w, T1w with contrast enhancement, T2w and FLAIR sequences. Other MRI sequences may work as well (just 
give it a try!)
- it was designed to be robust with respect to brain tumors, lesions and resection cavities
- it is very fast on GPU with <10s run time per MRI sequence. Even on CPU it is not slower than other commonly 
used tools
 
##Installation Instructions

1) Clone this repository
2) Go into the HD_BET directory and install with
    ```
    pip install -e .
    ```
3) Per default, model parameters will be downloaded to ~/.hd-bet_params. If you wish to use a different folder, open 
HD_BET/paths.py in a text editor and modify ```folder_with_parameter_files```


## How to use it
Using HD_BET is straightforward. We provide CPU as well as GPU support. Running on GPU is a lot faster though 
and should always be preferred. Here is a minimalistic example of how you can use HD-BET (you need to be in the HD_BET 
directory)

```bash
hd-bet -i INPUT_FILENAME
```

INPUT_FILENAME must be a nifti (.nii.gz) file containing 3D image data. 4D image sequences are not supported. 
INPUT_FILENAME can be either T1w, T1w with contrast agent, T2w or FLAIR MRI image. Other modalities might work as well.
Input images must match the orientation of MNI152! Use fslreorient2std <sup>1</sup> to ensure that is the case!

By default, this will run in GPU mode, use the parameters of all five models (which originate from a five-fold 
cross-validation), use test time data augmentation by mirroring along all axes and not do any postprocessing.

For batch processing it is faster to process an entire folder at once as this will mitigate the overhead of loading 
and initializing the model for each case:

```bash
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER
```

The above command will look for all nifti files (*.nii.gz) in the INPUT_FOLDER and save the brain masks under the same name
in OUTPUT_FOLDER.

To modify the device (CPU/GPU), whether to use test time data augmentation and postprocessing please refer 
to the documentation of run.py:

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



<sup>1</sup>https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained