import os
import json
import time
import boto3
from pathlib import Path
from typing import Optional, Union

class Config:
    def __init__(self):
        self.batch_size = 16 or 512
        self.name = "resnet50_imagenet_1k_onecycleLr"
        self.workers = 12
        self.max_lr = 0.175
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 2
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4
        self.train_folder_name =  '/Users/chiragtagadiya/MyProjects/EMLO_V4_projects/DVC-pytorch-lightning-MLOps/data/bird_small/train' or'/mnt/data/imagenetdata/train'
        self.val_folder_name = '/Users/chiragtagadiya/MyProjects/EMLO_V4_projects/DVC-pytorch-lightning-MLOps/data/bird_small/test' or '/mnt/data/imagenetdata/val'

    def __repr__(self):
        return str(self.__dict__)
    

class JsonLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        
    def log_metrics(self, epoch_metrics):
        self.metrics.append(epoch_metrics)
        with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

class CSVLogger:
    def __init__(self, log_dir, training_filename='training_log.csv', test_filename='test_log.csv'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        self.training_csv_file = os.path.join(log_dir, training_filename)
        self.test_csv_file = os.path.join(log_dir, test_filename)
        
    def log_metrics(self, epoch_metrics):
        self.metrics.append(epoch_metrics)
        
        # Determine which file to write to based on stage
        csv_file = self.training_csv_file if epoch_metrics.get('stage') == 'train' else self.test_csv_file
        
        # Write header if file doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as f:
                header = ','.join(epoch_metrics.keys())
                f.write(f"{header}\n")
        
        # Append the metrics
        with open(csv_file, 'a') as f:
            values = ','.join(str(v) for v in epoch_metrics.values())
            f.write(f"{values}\n")


def upload_file_to_s3(
    model_path: Union[str, Path],
    bucket_name: str,
    s3_prefix: str = 'imagenet_full/',
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
    aws_region: Optional[str] = None
) -> str:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    timestamp = int(time.time())
    timestamped_name = f"{model_path.stem}_{timestamp}{model_path.suffix}"
    
    aws_access_key = aws_access_key or os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = aws_secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = aws_region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not all([aws_access_key, aws_secret_key]):
        raise Exception("AWS credentials not found")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    s3_prefix = s3_prefix.rstrip('/') + '/'
    s3_key = f"{s3_prefix}{timestamped_name}"
    
    try:
        s3_client.upload_file(str(model_path), bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        raise Exception(f"Failed to upload model to S3: {str(e)}")