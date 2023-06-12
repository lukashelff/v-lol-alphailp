# alphaILP on V-LoL-Trains

This is the implementation of alphaILP extended to learn the V-LoL-Trains. The original alphaILP is proposed
in [Learning to Compose Dynamic Tree Structures for Visual Contexts](https://arxiv.org/abs/2004.00646). The original
code is available at https://github.com/ml-research/alphailp.git.

![ailp](./imgs/aILP.png)

## Set up the environment

```
docker build -t vlol-alpha-ilp .
```

## Training

### V-LoL-Trains Dataset:
Download the V-LoL-Trains dataset from [here](https://sites.google.com/view/
v-lol) and link the extracted folder in the docker command, e.g. `-v $(pwd)/V-LoL-dataset:/NSFR/data/michalski/all`. 
```
docker run --gpus device=0 --shm-size='20gb' --memory="700g" -v $(pwd)/v-lol-alphailp:/NSFR -v $(pwd)/V-LoL-dataset:/NSFR/data/michalski/all vlol-alpha-ilp python3 src/michalski_cross_val.py --dataset-type michalski --dataset theoryx --batch-size 10 --n-beam 50 --t-beam 5 --m 2 --device 0
```

# LICENSE

See [LICENSE](./LICENSE). The [src/yolov5](./src/yolov5) folder is following [GPL3](./src/yolov5/LICENSE) license.
