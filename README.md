# IRGAN

We  provide PyTorch implementations for IRGAN. 
Details of IRGAN can be seen in "Visible-to-infrared Image Translation Based on an Improved CGAN"(The Visual Computer,TVCJ).

#Train

python train.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction BtoA

#Test

python test.py --dataroot ./datasets/VEDAI --name VEDAI_IRGAN --model IRGAN --direction BtoA

#Acknowledgments

Our code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , https://github.com/NVIDIA/pix2pixHD and https://github.com/facebookresearch/ConvNeXt.
