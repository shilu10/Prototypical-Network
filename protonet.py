"""Implementation of prototypical networks for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils import tensorboard
from utils import score 


NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4

SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self, device):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            device (str): device to be used
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, device, compile=False, backend=None, learner=None, val_interval=None, save_interval=None, bio=False):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        self.device = device
        if learner is None:
            self._network = ProtoNetNetwork(device)
        else:
            self._network = learner.to(device)

        self.val_interval = VAL_INTERVAL if val_interval is None else val_interval
        self.save_interval = SAVE_INTERVAL if save_interval is None else save_interval
        self.bio = bio

        if(compile == True):
            try:
                self._network = torch.compile(self._network, backend=backend)
                print(f"ProtoNetNetwork model compiled")
            except Exception as err:
                print(f"Model compile not supported: {err}")

        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for i, task in enumerate(task_batch):
            # print(i)
            images_support, labels_support, images_query, labels_query = task
            # print(images_support.shape, labels_support.shape, images_query.shape, labels_query.shape)
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            ### START CODE HERE ###
            # TODO: finish implementing this method.
            # For a given task, compute the prototypes and the protonet loss.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate loss_batch, accuracy_support_batch, and
            # accuracy_query_batch.

            # compute prototypes without tracking gradients
                # Generate support features
            supp_feats = self._network(images_support)

            classes = torch.unique(labels_support, sorted=True)
            prototypes = []
            for c in classes:
                idxes = (labels_support == c).nonzero()
                class_feats = supp_feats[idxes]
                prototype = torch.mean(class_feats, dim=0)
                prototypes.append(prototype)

            prototypes = torch.cat(prototypes)  # (num_classes, feature_dim)

            # now to measure distances to all means
            # (batch_size, feature_dim)
            query_feats = self._network(images_query)
            # (batch_size, num_classes, feature_dim)
            query_feats = torch.stack([query_feats] * prototypes.size(0), dim=1)

            expanded_prototypes = prototypes.expand(*query_feats.shape)

            # (batch_size, num_classes, feature_dim)
            query_diffs = query_feats - expanded_prototypes
            # (batch_size, num_classes)
            query_sq_norms = torch.norm(query_diffs, p=2, dim=-1).square()
            query_logits = -query_sq_norms

            loss = F.cross_entropy(query_logits, labels_query)
            loss_batch.append(loss)

            # compute accuracies
            query_acc = score(query_logits, labels_query)
            accuracy_query_batch.append(query_acc)

            # for calculating acc and loss for support set

            # (batch_size, num_classes, feature_dim)
            supp_feats = torch.stack([supp_feats] * prototypes.size(0), dim=1)

            expanded_prototypes = prototypes.expand(*supp_feats.shape)

            # (batch_size, num_classes, feature_dim)
            supp_diffs = supp_feats - expanded_prototypes
            # (batch_size, num_classes)
            supp_sq_norms = torch.norm(supp_diffs, p=2, dim=2).square()
            supp_logits = -supp_sq_norms
            supp_acc = score(supp_logits, labels_support)
            accuracy_support_batch.append(supp_acc)

            ### END CODE HERE ###
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_meta_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        MAX_TRAIN = len(dataloader_meta_train)
        # exit()
        for i_step, task_batch in enumerate(
                dataloader_meta_train,
                start=self._start_train_step
        ):
            if i_step > MAX_TRAIN:
                break
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % self.val_interval == 0:
                print("Start Validation...")
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for i, val_task_batch in enumerate(dataloader_meta_val):
                        if self.bio and i > 600:
                            break
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                    ci95 = 1.96 * np.std(accuracies_query) / np.sqrt(600 * 4)
                if self.bio:
                    print(
                        f'Validation: '
                        f'loss: {loss:.3f}, '
                        f'support accuracy: {accuracy_support:.3f}, '
                        f'query accuracy: {accuracy_query:.3f}',
                        f'Ci95: {ci95:.3f}'
                    )
                else:
                    print(
                        f'Validation: '
                        f'loss: {loss:.3f}, '
                        f'support accuracy: {accuracy_support:.3f}, '
                        f'query accuracy: {accuracy_query:.3f}'
                    )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )
                if self.bio:
                    writer.add_scalar(
                        'val_accuracy/ci95',
                        ci95,
                        i_step
                    )
            if i_step % self.save_interval == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for i, task_batch in enumerate(dataloader_test):
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step, filename=""):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load
            filename (str): directly setting name of checkpoint file, default ="", when argument is passed, then checkpoint will be ignored

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        ) if filename == "" else filename
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


if __name__ == '__main__':
    network = ProtoNet(
            log_dir="logs/",
            learning_rate=0.001,
            device='cpu'
        )

    print(network)