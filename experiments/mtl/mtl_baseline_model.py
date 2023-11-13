import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import wandb

from drpm.diff_rand_part_model import RandPart


class MTLBaseline(pl.LightningModule):
    def __init__(
        self,
        tasks,
        loss_fn,
        models,
        n_partition_samples: int = 50,
        lr=1e-2,
        seed=1,
        regularization_weight: float = 0,
        regularization_strength=10.0,
        eval_noise_ratios=None,
        training_noise=0,
        device="cuda",
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        if not eval_noise_ratios:
            self.training_noise_suffix = ""
        elif training_noise == 0 or training_noise == 1:
            self.training_noise_suffix = f"_n{int(training_noise)}"
        else:
            self.training_noise_suffix = f"_n{training_noise}"

        self.eval_noise_ratios = eval_noise_ratios
        if eval_noise_ratios:
            self.noise_ratio_suffixes = list(
                map(
                    lambda x: f"_n{x}",
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                )
            )
        else:
            self.noise_ratio_suffixes = [""]

        self.net_rep = models[0](n_partition_samples)
        net_tasks = nn.ModuleDict()
        for k, t in enumerate(self.hparams.tasks):
            net_t = models[1][k](n_partition_samples)
            net_tasks[t] = net_t
        self.net_tasks = net_tasks

        # Store final rand score and nmi
        self.register_buffer("final_accuracies", torch.zeros(len(self.hparams.tasks)))
        self.register_buffer("final_mean_acc", torch.zeros(len(self.hparams.tasks)))

        # store num params
        self.num_tasks= len(self.hparams.tasks)

    def training_step(self, batch, batch_idx, optimizer_idx=1):
        # training_step defines the train loop.
        # it is independent of forward
        images = batch[0]
        labels = {}
        for k, t in enumerate(self.hparams.tasks):
            labels[t] = batch[k + 1]

        # Get current assignment matrix
        rep, out_tasks = self(images)

        losses_tasks = {}
        loss = torch.zeros(1, device=self.hparams.device)

        # Regularization to increase sparsity in latent representation
        if self.hparams.regularization_weight > 0:
            loss += self.hparams.regularization_weight * rep.abs().mean()

        for k, t in enumerate(self.hparams.tasks):
            loss_t = self.hparams.loss_fn[t](out_tasks[t], labels[t])
            losses_tasks[t] = loss_t
            loss += 1/float(self.num_tasks) * loss_t
            # logging task loss
            self.log("loss/train_loss_" + t, loss_t, prog_bar=True)

        self.log("loss/train_loss", loss, prog_bar=True)
        return loss

    def infer(self, x):
        inferred = self(x)[-1]
        return inferred

    def validation_step(self, batches, batch_idx):
        # After every epoch, log current rand_score

        if not self.eval_noise_ratios:
            batches = [batches]
        # Initaialize logger dict
        log = {}
        losses_tasks = {}
        acc_tasks = {}
        sparsity_n_zeros = dict()
        sparsity_n_zeros_neighbourhood = dict()

        for batch, suffix in zip(batches, self.noise_ratio_suffixes):

            images = batch[0]
            labels = {}
            for k, t in enumerate(self.hparams.tasks):
                labels[t] = batch[k + 1]

            # Compute rand score with mean partitioning
            rep, inf_tasks = self(images)
            log["sparsity/n_zeros" + suffix] = (rep == 0).float().sum(axis=-1).mean()
            log["sparsity/n_zeros_neighbourhood" + suffix] = (
                rep.isclose(torch.zeros(1, device=self.device), atol=1e-10)
                .float()
                .sum(-1)
                .mean()
            )

            sparsity_n_zeros[suffix] = log["sparsity/n_zeros" + suffix]
            sparsity_n_zeros_neighbourhood[suffix] = log[
                "sparsity/n_zeros_neighbourhood" + suffix
            ]

            for k, t in enumerate(self.hparams.tasks):
                t_suffix = t + suffix
                inf_t = inf_tasks[t].cpu()
                label_t = labels[t].cpu()
                loss_t = self.hparams.loss_fn[t](inf_t, label_t)
                pred_t = inf_t.data.max(1, keepdim=True)[1]
                acc_t = accuracy_score(label_t, pred_t)
                losses_tasks[t_suffix] = loss_t
                acc_tasks[t_suffix] = acc_t
                log["accuracy/accuracy_" + t_suffix] = acc_t
                log["loss/loss_" + t_suffix] = loss_t

            # Log validation metrics
        self.logger.log_metrics(log)
        return losses_tasks, acc_tasks, sparsity_n_zeros, sparsity_n_zeros_neighbourhood

    def validation_epoch_end(self, validation_outputs):
        loss_tasks = {}
        acc_tasks = {}
        accuracies = []

        for suffix in self.noise_ratio_suffixes:
            for k, t in enumerate(self.hparams.tasks):
                t_suffix = t + suffix
                loss_tasks[t_suffix] = []
                acc_tasks[t_suffix] = []

            sparsity_n_zeros = list()
            sparsity_n_zeros_neighbourhood = list()
            for out_batch in enumerate(validation_outputs):
                loss_tasks_batch = out_batch[1][0]
                acc_tasks_batch = out_batch[1][1]
                sparsity_n_zeros.append(out_batch[1][2][suffix])
                sparsity_n_zeros_neighbourhood.append(out_batch[1][3][suffix])
                for k, t in enumerate(self.hparams.tasks):
                    t_suffix = t + suffix
                    loss_tasks[t_suffix].append(loss_tasks_batch[t_suffix].item())
                    acc_tasks[t_suffix].append(acc_tasks_batch[t_suffix].item())

            self.log(
                "sparsity/avg_n_zeros" + suffix, torch.tensor(sparsity_n_zeros).mean()
            )
            self.log(
                "sparsity/avg_n_zeros_neighbourhood" + suffix,
                torch.tensor(sparsity_n_zeros_neighbourhood).mean(),
            )
            for k, t in enumerate(self.hparams.tasks):
                t_suffix = t + suffix
                mean_loss_t = torch.mean(torch.tensor(loss_tasks[t_suffix]))
                mean_acc_t = torch.mean(torch.tensor(acc_tasks[t_suffix]))
                self.log("loss/avg_val_loss_" + t_suffix, mean_loss_t)
                self.log("accuracy/avg_acc_" + t_suffix, mean_acc_t)

                if suffix == self.training_noise_suffix:
                    accuracies.append(mean_acc_t)

        mean_task_acc = sum(accuracies) / len(accuracies)
        self.log("accuracy/mean_tasks_acc", mean_task_acc)
        # Save current final scores
        self.final_accuracies = torch.tensor(accuracies)
        self.final_mean_acc = mean_task_acc

    def forward(self, images):
        """
        Return a partition matrix that assigns each dimension in x to a cluster
        """

        rep, _ = self.net_rep(images, None)

        out_tasks = {}
        for k, t in enumerate(self.hparams.tasks):
            out_t, _ = self.net_tasks[t](rep, None)
            out_tasks[t] = out_t
        return rep, out_tasks

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99),
        }
