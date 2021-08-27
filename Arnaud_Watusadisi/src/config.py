"""Add all arguments to this file"""
import argparse

args = argparse.Namespace(
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 3072,

    # Number of training epochs
    num_epochs = 500,

    # Learning rate for optimizers
    lr = 0.001,

    # Number of hidden_size.
    hidden_size = 10,

    # Number of out_size.
    out_size=10,
    
    ckpt_path = "checkpoints"
    
)