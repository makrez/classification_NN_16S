# classification_NN_16S

## Dependencies

This repository uses the docker image `docker://makrezdocker/ml-16s:1.0`. Then,
a singularity image is build with:

```
singualriry pull docker://makrezdocker/ml-16s:1.0;
```

## runnin the container

Log in to a node with GPU support:

```
srun --pty  --nodelist <node> -c 2 --mem=4000 --time=02:00:00 /bin/bash
```

Test if GPU is available:

```
singularity exec --nv --bind /data ../singularity_images/ml-16s_1.0.sif python ./check_gpu.py
```
