from beam import App, Runtime, Image, Volume, VolumeType

training_app = App(
    name="train_qa_ray",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10", 
            python_package="requirements.txt"
        ),
    ),
    volumes=[
        Volume(path="./qa_dataset_ray", name="qa_dataset_ray"),
        Volume(
            path="./output",
            name="train_qa_ray_output",
            volume_type=VolumeType.Persistent,
        ),
        Volume(
            path="./model_cache",
            name="model_cache",
            volume_type=VolumeType.Persistent
        ),
    ],
)


@training_app.run()
def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    env_file_path: str =  ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
):
    """
    Trains machine learning model using the specified configuration file and dataset.

    Args:
        config_file (str): Path to the configuration file for the training process.
        output_dir (str): Directory where the trained model will be saved to.
        dataset_dir (str): Directory where the training dataset is located.
        env_file_path (str, optional): Path to the env file. Defaults to ".env".
        logging_config_path (str, optional): Path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str, optional): Directory where the trained model will be cached. Defaults to None.
    """
