# BASELINE

***

This repo contains some baseline for public datasets.

### Requirement

* Python 3.6+

* Cuda 9.2+ or 10.1+
* Package installed requirement.txt

```bash
pip install requirement.txt
```

### Directory

Your directory should look like below after **git clone**.  

```
.
├── data
├── requirement.txt
├── saved
├── scripts
├── src
└── tags
```

Create directory **models** and directory **logdirs** under **saved** directory.

## Dataset.

| #Dataset | #Supported | #Train |  #Val  |                 #Mean ,  STD                 |
| :------: | :--------: | :----: | :----: | :------------------------------------------: |
|  MNIST   |     Y      | 50,000 | 10,000 |              [0.131],  [0.308]               |
| CIFAR10  |     Y      | 50,000 | 10,000 | [0.502, 0.494, 0.461], [0.249, 0.246, 0.263] |
| CIFAR100 |     Y      | 50,000 | 10,000 | [0.505, 0.488, 0.442], [0.267, 0.256, 0.276] |
|   SVHN   |     Y      | 73,257 | 26,032 | [0.437, 0.441, 0.470] [0.200, 0.203, 0.199]  |



### Run

Run your scripts in root directory.

```bash
bash ./scripts/CUB_000.sh
```

### Logs

You can observe logs in three ways

* Standard output.
* **tail -f** Logs file, eg. **tail -f 000.log** under saved/logdirs/000/
* Tensorboard, which also placed under saved/logdirs/000/
