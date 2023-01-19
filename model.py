""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

### from tensorflow.keras.models import load_model
from torchvision import models
import torch
def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='COCO_WITH_VOC_LABELS_V1',progress=True)
    ### model = models.segmentation.deeplabv3_mobilenet_v3_large()
    #model=torch.load('./exp_dir/mnv3l.pt')
    model.classifier = DeepLabHead(960, outputchannels)
    # Set the model in training mode
    model.train()
    return model


# models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
#progress=True