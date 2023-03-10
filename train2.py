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
    train_loader = DataLoader(augmented_train_dataset, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    val_loader = DataLoader(val_dataset, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)


    # Model definition
    # UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)  # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    
    net = SiameseNetwork(backbone="resnet50")
        
    if Flags.load_model:
        net.load_state_dict(torch.load(Flags.load_model_path))
    if use_gpu:
        net.cuda()


    # Training
    # Training setup
    net.train()  # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr)
    optimizer.zero_grad()  # Gradient initialization
    train_loss = []
    loss_val = 0
    time_start = time.time()
    accuracy = []

    # Training loop
    for batch_id, (img1, img2, label) in enumerate(train_loader, start=1):
        if batch_id > max_iter:
            break
        if use_gpu:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)

        optimizer.zero_grad()  # Gradient reset per batch

        # Prediction and error estimation
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        train_loss.append(loss_val)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch_id % Flags.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            batch_id, loss_val / Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()

        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id + 1) + ".pt")

        if batch_id % Flags.test_every == 0:
            net.eval()
            right, error = 0, 0
            for val_id, (test1, test2) in enumerate(val_loader, 1):
                if use_gpu:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)

                # Prediction and error estimation
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else:
                    error += 1

            acc = right * 1.0 / (right + error)
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
            batch_id, right, error, acc))
            print('*' * 70)
            accuracy.append(acc)
            net.train()

    #  learning_rate = learning_rate * 0.95

    # Save loss time series
    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    # Accuracy metrics
    acc = 0.0
    for d in accuracy:
        acc += d
    print("#" * 70)
    print("mean accuracy: ", acc / len(accuracy))
    print("max accuracy: ", max(accuracy))
    print("#" * 70)
    train_dataset   = Dataset(args.train_path, shuffle_pairs=True, augment=True)
    val_dataset     = Dataset(args.val_path, shuffle_pairs=False, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=8)

    model = SiameseNetwork(backbone=args.backbone)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    best_val = 3

    for epoch in range(args.epochs):
        print("[{} / {}]".format(epoch, args.epochs))
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

        #writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        #writer.add_scalar('train_acc', correct / total, epoch)

        #print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
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
        #writer.add_scalar('val_loss', val_loss, epoch)
        #writer.add_scalar('val_acc', correct / total, epoch)

        #print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )            

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )
