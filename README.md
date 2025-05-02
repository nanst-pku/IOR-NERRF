# An Index of Refraction Adaptive Neural Refractive Radiance Field for Transparent Scenes
------

An improvement of NeRRF: 3D Reconstruction and View Synthesis for Transparent and Specular Objects with Neural Refractive-Reflective Fields
------

We borrowed the code from the [NeRRF](https://github.com/JunchenLiu77/NeRRF) author and made modifications based on it.

### Setup

```
conda env create -f environment.yml
conda activate nerrf
```

### Dataset

Our blender synthetic dataset can be found here: [blender_dataset](https://drive.google.com/drive/folders/1us6geRhh0FwCoXy7VQAzixP2z1JF6QtS?usp=sharing)

### Usage

1. In the `NeRRF ` directory, ensure you have the following data:

    ```
    NeRRF
    |-- data
        |-- blender
            |-- depth
            |-- meta
            |-- specular
            |-- transparent
    ```

2. Run the training script

    ```
    # reconstruct the specular horse of the blender dataset
    sh trace_scripts/train_horse_t.sh
    ```

    + The scripts can be switched from the geometry reconstruction stage to the radiance estimating stage by changing the value of the `stage` variable. First set `stage` to `1` for geometry reconstruction. Then set `stage` to `2` for radiance field reconstruction.
    + For the synthetic dataset, edit `NeRRFsngp.conf` to modify the training parameters. For the real synthetic dataset, edit `NeRRFngp.conf` to modify the training parameters.

3. Run the evaluation script

    ```
    # evaluate the specular horse of the blender dataset
    sh trace_scripts/train_horse_t.sh
    ```
