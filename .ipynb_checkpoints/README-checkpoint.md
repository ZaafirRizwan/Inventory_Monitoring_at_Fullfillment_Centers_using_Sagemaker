

# Inventory_Monitoring_at_Fullfillment_Centers_using_Sagemaker

Fulfillment Centers have bin locations for storing goods as it simplifies inventory management
processes. Robots often move objects as part of their operations. Objects are carried to bin locations
which can contain more than one object. Whole purpose of these bin locations is to organise the
objects so they can be easily retrieved when searched.

## Dataset

### Overview
Datasets which will be used for this project is Amazon Bin Image Dataset. This dataset contains around
500,000 images along with the metadata of images. Also, the metadata included in the dataset
contains number of objects in the bin, its dimension, and the type of object. Link to the dataset
https://registry.opendata.aws/amazon-bin-imagery/. However,here not all data was used for training

### Access
First, a subset of data was downloaded from this link(https://registry.opendata.aws/amazon-bin-imagery/). Then, it was copied to s3 bucket and during training it was fetched from the bucket

## Model Training
I used pretrained resnet-18 model. Model Hyperparameters were:
1. batch_size
2. epochs
3. Learning Rate

As mentioned earlier, i only used subset of data so i used pretrained resnet architecture as the backbone and at the end, I appended fully connected layers for prediction.

## Machine Learning Pipeline
Data was first downloaded and uploaded to s3.Then an estimator function was defined for training of the model. Once the model is trained, it is deployed to an Endpoint.


## Standout Suggestions
**HP Tuning** Here I created a hp tuner object where mainly 3 hyperparameters were tuned. Batch_size,learning_rate and epochs.

**Debugging and Profiling** I defined debugging and profiling rules in this step as they help in improving model performance.

**Model Deployment** I deployed the model to and enpoint and invoked it for inference.

**Multi-Instance Training** Here, I set the instance_count to 2 for multi_instance training.
