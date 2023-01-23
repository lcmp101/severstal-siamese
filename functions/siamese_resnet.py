import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from keras.applications import ResNet50


class Siamese(nn.Module):
# https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
# https://keras.io/examples/vision/siamese_contrastive/
    def __init__(self, pretrained=None, classify=False, embs=False, num_classes=None, num_out_comp=None, dropout_prob=0.6):
        super(Siamese, self).__init__()
        
        if pretrained == True:
            temp_weights = "imagenet"
        else
            temp_weights = None
        
        
        # Define the tensors for the two input images
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        
        model = ResNet50(include_top = True,
            weights = temp_weights,
            input_tensor = None,
            input_shape = None,
            pooling = None,
            classes = num_classes)
        
        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

        # return the model
        return siamese_net


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home
