# Adapted from lightly tutorial https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simsiam_esa.html

# module import
import torch
import torch.nn as nn
import lightly.data as data
from lightly.models.modules.heads import SimSiamPredictionHead
from lightly.models.modules.heads import SimSiamProjectionHead
from torch.utils.data import DataLoader, random_split
import torchvision
from lightly.loss import NegativeCosineSimilarity
import math

# set parameters
data_path = 'PATH/TO/DATA/'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# set hyperparameters

momentum = XXX
batch_size = XXX
max_epochs = XXX

# dimension of the embeddings - fixed by choice of backbone
num_ftrs = 512
# dimension of the output of the prediction and projection heads
out_dim = proj_hidden_dim = 512
# the prediction head uses a bottleneck architecture
pred_hidden_dim = 128

# Model Class
class SimSiam(nn.Module):
    def __init__(
        self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p


# define collate function to apply random augmentations to the input images
collate_fn = data.ImageCollateFunction(
    input_size=256,         # set input size of image
    cj_prob = 0.,           # probability of color jitter application 
    cj_bright = 0.,         # how much to change brightness
    cj_contrast = 0.,       # how much to change contrast
    cj_sat = 0.,            # how much to change saturation
    cj_hue = 0.,            # how much to change hue
    min_scale = 0.3,        # minimum size of cropped images [0, 1]
    random_gray_scale = 0., # probability of gray scale
    gaussian_blur = 0.7,    # probability of gaussian blur application
    kernel_size = 0.025,    # size of gaussian blur kernel
    vf_prob = 0.7,          # probability of vertical flips/mirror
    hf_prob = 0.7,          # probability of horizonal flips/mirror
    rr_prob = 0.,           # probability of random rotation
    normalize = False       # normalization of images
    )

# create dataset from input folder
dataset = data.LightlyDataset(input_dir = data_path)

# split into train/val
training_set, validation_set = random_split(dataset, [0.7, 0.3])

training_dataloader = DataLoader(
    training_set,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    drop_last = True)

validation_dataloader = DataLoader(
    validation_set,
    batch_size = batch_size,
    shuffle = False,
    collate_fn = collate_fn,
    drop_last = True)

# define backbone and create model
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)

# define loss function
criterion = NegativeCosineSimilarity()

# define learning rate - scale with batch size
lr = 0.05 * batch_size / 256

# define optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
    weight_decay=5e-4
)

avg_loss = 0.; val_avg_loss = 0.
avg_output_std = 0.; val_avg_output_std = 0.
for e in range(max_epochs):

    for (x0, x1), _, _ in training_dataloader:

        # move images to the device
        x0 = x0.to(device)
        x1 = x1.to(device)

        # run model on both augmentations, get projections (z0 and z1) and predictions (p0 and p1)
        z0, p0 = model(x0)
        z1, p1 = model(x1)

        # calculate loss
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        # backpropagation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculate the mean per-dimension standard deviation of the outputs for monitoring collapse
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

    for (x0, x1), _, _ in validation_dataloader:

        # calculate loss on validation set

        x0 = x0.to(device)
        x1 = x1.to(device)

        z0, p0 = model(x0)
        z1, p1 = model(x1)

        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))

        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        w = 0.9
        val_avg_loss = w * val_avg_loss + (1 - w) * loss.item()
        val_avg_output_std = w * val_avg_output_std + (1 - w) * output_std.item()

    # calculate collapse
    collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
    val_collapse_level = max(0., 1 - math.sqrt(out_dim) * val_avg_output_std)

    # print intermediate results
    print(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')

# save model to use later
torch.save(model.state_dict(), 'SSL.pt')

