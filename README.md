<h1 align="center">
  <br>
  dockerized-finetuning 🐳
  <br>
</h1>

[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/klfx/dockerized-finetuning/main)](https://github.com/klfx/dockerized-finetuning/commits/main)
[![Docker Pulls](https://img.shields.io/docker/pulls/klfxch/dockerized-finetuning)](https://hub.docker.com/r/klfxch/dockerized-finetuning)
[![Docker Image Size (tag)](https://img.shields.io/docker/image-size/klfxch/dockerized-finetuning/latest)](https://hub.docker.com/r/klfxch/dockerized-finetuning)

dockerized-finetuning is a finetuning project for distilbert-base-uncased (on glue mrpc) with integrated Weights&Biases experiment tracking. Utilizing containerized training offers advantages such as simplified deployment, scalability, and platform independence. The processing and training is implemented with PyTorch Lightning and supports CUDA by default. Disclaimer: This was created as part of a college AI course project. 

* 👾 Finetuning for distilbert-base-uncased on GLUE MRPC
* 📊 Experiment tracking with Weights&Biases
* 🔧 Customize to your needs 


## Download

### Docker Image
This is the fastest way to get started and try out stuff. The image of the latest build can be pulled from Docker Hub.

```docker pull klfxch/dockerized-finetuning```

### Build
To build your own docker image, clone this repository and build the image with the following command. 

```docker build -t [your_image_name] .```

## Run Instructions

To run the project with interactive console(`-it`) and CUDA passthrough(`--gpus all`), run the following command.

```docker run -it --gpus all dockerized-finetuning [ARGUMENTS]```

The arguments allow to customize the training process and experiment tracking. For more information about wandb arguments, see <a>https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html</a>.

| Options          | Description                             |
|------------------|-----------------------------------------|
| --checkpoint_dir | Directory to save checkpoints (Docker internal)|
| --learning_rate  | Learning rate                           |
| --warmup_steps   | Warmup steps                            |
| --weight_decay   | Weight decay                            |
| --epochs         | Max Epochs                              |
| --disable_wandb  | Disable all tracking with Weight&Biases |
| --wandb_entity   | Weight&Biases entity (username or team) |
| --wandb_project  | Weight&Biases project name              |
| --wandb_api_key  | Weight&Biases API key                   |
| --wandb_log_model| Weight&Biases Log Model, defines how many checkpoints to log |



## Examples

### (1) Interactive Run with custom hyperparameters and tracking

```docker run -it --gpus all dockerized-finetuning --learning_rate 1e-5 --epochs 10 --wandb_project [your_project] --wandb_api_key [your_api_key]```

### (2) Unattended Run
Note: For unattended runs with wandb, you need to specify `--wandb_api_key`

```docker run --gpus all dockerized-finetuning --wandb_project [your_project] --wandb_api_key [your_api_key]```

### (3) Unattended Offline Run without tracking

```docker run --gpus all dockerized-finetuning --disable_wandb```

### (4) Unattended Run and upload all checkpoints to wandb
Note: This may consume a lot of storage space on wandb.

```docker run --gpus all dockerized-finetuning --wandb_project [your_project] --wandb_api_key [your_api_key] --wandb_log_model all```

## FAQ


### Does this project require CUDA to run?

No, this will run fine on a CPU. However, it will be much slower. If you have a CUDA enabled GPU, remember to pass it to the Docker Container. In my case, using a GPU reduced training time from 50 minutes to 5 minutes (with default parameters).

### Does the container need access to the internet?

Yes, the container will download datasets from HuggingFace during initialization. 

### How can I access my model checkpoints?

The simplest method to capture checkpoints is to specify the `--wandb_log_model` flag. This will automatically upload checkpoints to your wandb project.

If you prefer to access your checkpoints locally, you can use Docker Bind mounts to access the internal directory. Mounts can be specified in the `docker run` command. (<a>https://docs.docker.com/storage/bind-mounts/</a>)

