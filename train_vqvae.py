# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import numpy as np
import torch
from VGG import *
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, FashionMNIST, CIFAR10
from vqvae import VQVAE
import options
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    # kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'none'}
    # # Define input options
    kwargs = {}
    parser = options.define_args(filename="train_vqvae", description='Replay effect')
    parser = options.add_general_options(parser, **kwargs)
    # parser = options.add_eval_options(parser, **kwargs)
    # parser = options.add_task_options(parser, **kwargs)
    # parser = options.add_model_options(parser, **kwargs)
    # parser = options.add_train_options(parser, **kwargs)
    # parser = options.add_replay_options(parser, **kwargs)
    # parser = options.add_bir_options(parser, **kwargs)
    # parser = options.add_allocation_options(parser, **kwargs)
    # # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    # options.set_defaults(args, **kwargs)
    # options.check_for_errors(args, **kwargs)
    return args
torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")


# Initialize model.
device = torch.device("cuda:0")
use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
    
}
general_args = {'dataset':'MNIST',
                'epochs' : 7,
                 'eval_every' : 100}
datasets = {
    'CIFAR100':CIFAR100,
    'CIFAR10': CIFAR10,
    'MNIST': FashionMNIST
}
model = VQVAE(**model_args).to(device)

# Initialize dataset.
batch_size = 32
workers = 10
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
data_root = "../data"
train_dataset = datasets[general_args['dataset']](data_root, True, transform, download=True)
train_data_variance = np.var(train_dataset.data / 255)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
)

# Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
# Learning".
beta = 0.25

# Initialize optimizer.
train_params = [params for params in model.parameters()]
lr = 3e-4
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

# Train model.

best_train_loss = float("inf")
model.train()
for epoch in range(general_args['epochs']):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = train_tensors[0].to(device)
        out = model(imgs)
        recon_error = criterion(out["x_recon"], imgs) / train_data_variance
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % general_args['eval_every']) == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            total_train_loss /= n_train
            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss

            print(f"total_train_loss: {total_train_loss}")
            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0

# Generate and save reconstructions.
model.eval()

valid_dataset = CIFAR100(data_root, False, transform, download=True)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=workers,
)

with torch.no_grad():
    for valid_tensors in valid_loader:
        break

    save_img_tensors_as_grid(valid_tensors[0], 4, "true")
    save_img_tensors_as_grid(model(valid_tensors[0].to(device))["x_recon"], 4, "recon")


# Train VGG 
model = VGG16_NET() 
model = model.to(device=device) 
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr) 


for epoch in range(general_args['epochs_vgg']): #I decided to train the model for 50 epochs
    loss_var = 0
    
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores,labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        if idx%64==0:
            print(f'Epoch [{epoch+1}/{general_args['epochs_vgg']}] || Step [{idx+1}/{len(train_loader)}] || Loss:{loss_var/len(train_loader)}')
    print(f"Loss at epoch {epoch+1} || {loss_var/len(train_loader)}")

    with torch.no_grad():
        correct = 0
        samples = 0
        for idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum()
            samples += preds.size(0)
        print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")