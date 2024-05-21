# GS3D-Segment
This Repo contains the implementation of GS3DSeg build on top of Nerfstudio. Given a set of Images with Transforms (with or without pointcloud data) GS3DSeg can quickly build 3D model of the scene with guassian splating with view-consistant object Segmentation



## Registering GS3DSeg with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running GS3DSeg
Run SAM.py once initally to get Mask for the Images

This repository creates a new Nerfstudio method named "GS3DSeg". To train with it, run the command:
```
ns-train GS3DSeg --data [PATH]
```

