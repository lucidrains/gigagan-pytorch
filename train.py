# Model Design
#
# This is a model design to do 2x, 4x, 8x upscaling.

import webdataset as wds
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig, Checkpoint
from ray.train.huggingface import AccelerateTrainer
from ray.air import session
from accelerate import Accelerator
from gigagan_pytorch import VisionAidedDiscriminator, UnetUpsampler, Discriminator
import torch
import torch.nn as nn
from lion_pytorch import Lion
import lpips
from torchvision import transforms
import random

# hyper-parameters
# Small image size(Concatted as a channel)
dim_text_feature = 64
num_epochs = 128

use_GLU_discrim = True

NUM_WORKERS = 16

l_local_size = 120
l_global_size = 8

w_size = l_local_size + l_global_size

# 2x upscaling
input_image_size = 128
image_size = 256

model_opts = {
    "2xUpscaling": {
        "image_size": image_size,
        "input_image_size": input_image_size,
    },
    "4xUpscaling": {
        "image_size": image_size
    }
}


class GigaGANTextConditionedUpscaler(nn.Module):
    def __init__(self):
        super(GigaGANTextConditionedUpscaler, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.vision_aided_discriminator = VisionAidedDiscriminator()
        self.discriminator = Discriminator(use_glu=use_GLU_discrim)

        upsampler_dim = 128

        self.upsampler = UnetUpsampler(
            dim=upsampler_dim,
            cross_attention=True,
            l_global_size=l_global_size,
            l_local_size=l_local_size,
            **model_opts
        )

    def loss(self, outputs):
        # Compute both discriminators for loss
        clip_loss = self.model.vision_aided_discriminator(outputs)
        discrim_loss = self.model.discriminator(outputs)
        plip_loss = self.model.lpip_loss(outputs)

        plips_loss_strength = 0.4
        clip_loss_strength = 0.4
        discrim_loss_strength = 0.2

        loss = plips_loss_strength * plip_loss +  \
            clip_loss * clip_loss_strength + \
            discrim_loss_strength * discrim_loss

        return loss

    def forward(self, images_in, text):
        out = self.upsampler(images_in, text)

        return out


# Unused
# class GigaGANUnconditionedUpscaler(nn.Module):
#     def __init__(self):
#         super(GigaGANTextConditionedUpscaler, self).__init__()
#         self.lpips_loss = lpips.LPIPS(net='alex')
#         self.vision_aided_discriminator = VisionAidedDiscriminator()
#         self.discriminator = Discriminator()

#         upsampler_dim = 128
#         input_size = 64
#         output_size = 256

#         self.upsampler = UnetUpsampler(
#             dim=upsampler_dim,
#             image_size=output_size,
#             input_image_size=input_size,
#         )

#     def forward(self, images_in):
#         # Concatenate the text embeddings with the images along the channel dimension
#         out = self.upsampler(images_in)

#         return out


# Takes crops of the images, prepares them for the upscaling task
class CroppedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Create a list to store crops and their labels
        labels = []

        # Define the range of possible base sizes for 64->256
        in_pairs = [(64, 64),
                    (64, 128),
                    (128, 64),
                    (128, 128),
                    (128, 256),
                    (256, 128),
                    (256, 256)]

        upscale_size = 4.0

        # Select random
        in_size = random.choice(in_pairs)
        out_size = in_size * upscale_size

        # Define the grid size for cropping

        # Use a FiveCrop transform to get crops from all corners and the center
        five_crop = transforms.FiveCrop(out_size)
        five_crops = five_crop(image)

        # Resize the crops
        resize = transforms.Resize((1/upscale_size, 1/upscale_size))
        resized_crops = [resize(crop) for crop in five_crops]

        # Create labels for each crop
        for i, (crop, resized) in enumerate(zip(five_crops, resized_crops)):
            crop_name = f"crop: {['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'][i]}"
            w, h = in_size
            ow, oh = out_size

            crop_label = f"{label} base: {w}x{h}, final: {ow}x{oh}, {crop_name}"

            labels.append(crop_label)

        return resized_crops, five_crops, labels

    def __len__(self):
        return len(self.dataset)


def train_wds(url):
    def distributed_training_loop():
        # Model and optimizer
        model = GigaGANTextConditionedUpscaler()

        opt_gen = Lion(model.generator.parameters(),
                       lr=1e-4, weight_decay=1e-2)

        opt_disc = torch.optim.Adam(
            model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Initialize the Accelerator
        accelerator = Accelerator()

        # Fetch training set from the session
        dataset_shard = session.get_dataset_shard("train")

        model, opt_gen, opt_disc = accelerator.prepare(
            model, opt_gen, opt_disc)

        for epoch in range(num_epochs):
            for batches in dataset_shard.iter_torch_batches(
                batch_size=32, dtypes=[torch.float, torch.float]
            ):
                x, y, text = batches
                # Train model here
                # model outputs
                outputs = model(x, text)

                loss = model.loss(y, outputs)

                # Zero out grads, do backward, and update optimizer
                opt_disc.zero_grad()
                opt_gen.zero_grad()
                accelerator.backward(loss)
                opt_disc.step()
                opt_gen.step()

                # Print what's happening with loss per 30 epochs
                if epoch % 20 == 0:
                    print(f"epoch: {epoch}/{num_epochs}, loss: {loss:.3f}")

            # Report and record metrics, checkpoint model at end of each
            # epoch
            session.report(
                {"loss": loss.item(), "epoch": epoch},
                checkpoint=Checkpoint.from_dict(
                    dict(
                        epoch=epoch,
                        model=accelerator.unwrap_model(model).state_dict(),
                    )
                ),
            )

    # Define scaling and run configs
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS, use_gpu=True)
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

    # Load dataset
    train_dataset = wds.WebDataset(url).shuffle(
        1000).decode("rgb").to_tuple("png", "json")

    trainer = AccelerateTrainer(
        train_loop_per_worker=distributed_training_loop,
        accelerate_config={},
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset},
    )

    result = trainer.fit()

    return result
