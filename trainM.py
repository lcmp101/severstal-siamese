from functions.utils import fixed_image_standardization
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
from functions import createTrain, createTest, splitDataSet, Siamese
import pickle
import time
import sys
import gflags

from siamese import SiameseNetwork
#from libs.dataset import Dataset

if __name__ == '__main__':

    # Parameters
    Flags = gflags.FLAGS
    gflags.DEFINE_string("data_path", "./train", "training folder")
    gflags.DEFINE_string("model_path", "models", "path to store model")
    gflags.DEFINE_bool("load_model", False, "Whether load a pretrained model or not")
    gflags.DEFINE_string("load_model_path", "models/model.pt", "pathname to load model")
    gflags.DEFINE_integer("imHeight", 105, "Image height")
    gflags.DEFINE_integer("imWidth", 105, "Image wifth")
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 32, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_bool("colab", False, "If Colab is used, use a reduced number of max_iter")
    gflags.DEFINE_integer("max_iter_colab", 5000, "number of iterations before stopping if using Colab")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")
    gflags.DEFINE_bool("save_errors", True, "Whether save val errors or not")

    Flags(sys.argv)

    # Process arguments
    os.makedirs(Flags.model_path, exist_ok=True)      # Creates model folder if not available
    resize_image = [Flags.imHeight, Flags.imWidth]
    if Flags.colab:
        max_iter = Flags.max_iter_colab
    else:
        max_iter = Flags.max_iter


    # Determine if an nvidia GPU is available
    use_gpu = torch.cuda.is_available()
    print('Using GPU: {}'.format(use_gpu))


    # Define dataset, data augmentation, and dataloader
    # Dataset splitting
    percentage = 0.8  # Percentage of samples used for training
    train_path, test_path = splitDataSet(Flags.data_path, percentage)
    
    # # # # # # # # # # # # # # # # # # # # # # # # Transformations # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Train
    train_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    # Data augmentation
    horz_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        transforms.RandomHorizontalFlip(p=1),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    vert_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        transforms.RandomVerticalFlip(p=1),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ]) 
    horz_vert_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
     ])
    rot1_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        transforms.RandomRotation(degrees=(0, 180)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    rot2_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        transforms.RandomRotation(degrees=(180, 360)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])     
    # Test
    test_transforms = transforms.Compose([
        transforms.Resize(resize_image),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = createTrain(train_path, transform=train_transforms)
    horz_dataset = createTrain(train_path, transform=horz_transforms)
    vert_dataset = createTrain(train_path, transform=vert_transforms)
    horz_vert_dataset = createTrain(train_path, transform=horz_vert_transforms)
    rot1_dataset = createTrain(train_path, transform=rot1_transforms)
    rot2_dataset = createTrain(train_path, transform=rot2_transforms)
    
    augmented_train_dataset = torch.utils.data.ConcatDataset([train_dataset,horz_dataset,vert_dataset,rot1_dataset,rot2_dataset])
    val_dataset = createTest(test_path, transform=test_transforms, times=Flags.times, way=Flags.way)
    
    # Dataloader
    train_dataloader = DataLoader(augmented_train_dataset, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)

    print("train loader length:", len(train_dataloader))
    print("val loader length:",len(val_dataloader))

    # Model definition
    # UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)  # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    
    model = SiameseNetwork(backbone="resnet50")   
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Flags.lr)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(Flags.model_path, "summary"))

    best_val = 10000000000

    epochs = 5

    for epoch in range(epochs):
        print("[{} / {}]".format(epoch, epochs))
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), y, (class1, class2) in train_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('train_acc', correct / total, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2) in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        val_loss = sum(losses)/max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": resnet50,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(Flags.model_path, "best.pth")
            )            

        save_after = 2
        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": resnet50,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )
