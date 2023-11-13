import math

import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import wandb

from drpm.diff_rand_part_model import RandPart


def get_initializer(initializer_name):
    initializer = None
    if initializer_name == "zeros":
        initializer = nn.init.zeros_
    elif initializer_name == "normal":
        initializer = lambda x: nn.init.normal_(x, 0, 1)
    else:
        raise ValueError(f"Unknown initializer '{initializer_name}'")

    return initializer


class DRPMmtl(pl.LightningModule):
    def __init__(
        self,
        tasks,
        loss_fn,
        models,
        n_partition_samples,
        n_clusters,
        lr=1e-2,
        seed=1,
        task_split_layer="dense",
        learn_const_omegas=False,
        learn_const_scores=False,
        const_omegas_initializer="normal",
        const_scores_initializer="normal",
        reverse_partitions_order=False,
        resample=True,
        hard=True,
        final_temp=1e-2,
        start_temp=1.0,
        num_steps_annealing=100,
        annealing_type="linear",
        regularization_partition_index=None,
        regularization_weight=0,
        drpm_use_encoding_model: bool = False,
        eval_noise_ratios=None,
        training_noise=0,
        device="cuda",
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Initialize random partition module
        self.rpm = RandPart(n_cluster=n_clusters, device=self.hparams.device)

        # gumbel softmax
        self.resample = resample
        self.hard = hard

        # temperature annealing
        self.final_temp = torch.tensor(final_temp)
        self.start_temp = torch.tensor(start_temp)
        self.num_steps_annealing = num_steps_annealing

        self.drpm_use_encoding_model = drpm_use_encoding_model
        self.relu = torch.nn.ReLU()

        self.drpm_encoding_model = None
        if drpm_use_encoding_model:
            self.drpm_encoding_model = models[0](n_partition_samples)
        else:
            self.drpm_encoding_model = nn.Identity()

        self.noise_ratio_suffixes = None
        self.eval_noise_ratios = eval_noise_ratios

        if not eval_noise_ratios:
            self.training_noise_suffix = ""
        elif training_noise == 0 or training_noise == 1:
            self.training_noise_suffix = f"_n{int(training_noise)}"
        else:
            self.training_noise_suffix = f"_n{training_noise}"

        if eval_noise_ratios:
            self.noise_ratio_suffixes = list(
                map(
                    lambda x: f"_n{x}",
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                )
            )
        else:
            self.noise_ratio_suffixes = [""]

        # log omega parameter (P(n|z))
        if learn_const_omegas:
            self.log_omega = nn.Parameter(
                torch.empty(size=(1, n_clusters), device=device, requires_grad=True)
            )
            initializer = get_initializer(const_omegas_initializer)
            with torch.no_grad():
                initializer(self.log_omega)
            self.estimate_log_omega = lambda x: self.log_omega

        else:
            if task_split_layer == "conv":
                self.estimate_log_omega = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        n_partition_samples * 4 * 4, n_clusters
                    ),  # FIXME: Assumes MultiMNIST
                )
            elif task_split_layer == "dense":
                self.estimate_log_omega = nn.Linear(n_partition_samples, n_clusters)

        # scores for (P(Y|n,z))
        if learn_const_scores:
            self.log_score = nn.Parameter(
                torch.empty(
                    size=(1, n_partition_samples), device=device, requires_grad=True
                )
            )
            initializer = get_initializer(const_scores_initializer)
            with torch.no_grad():
                initializer(self.log_score)
            self.estimate_log_score = lambda x: self.log_score

        else:
            if task_split_layer == "conv":
                self.estimate_log_score = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        n_partition_samples * 4 * 4, n_partition_samples
                    ),  # FIXME: Assumes MultiMNIST
                )
            elif task_split_layer == "dense":
                self.estimate_log_score = nn.Linear(
                    n_partition_samples, n_partition_samples
                )

        # reverse sampling order
        self.task_keys = sorted(self.hparams.tasks)
        if reverse_partitions_order:
            self.task_keys = list(reversed(self.task_keys))

        self.net_rep = models[0](n_partition_samples)  #
        net_tasks = nn.ModuleDict()
        for k, t in enumerate(self.task_keys):
            net_t = models[1][k](n_partition_samples)
            net_tasks[t] = net_t
        self.net_tasks = net_tasks

        if annealing_type == "linear":
            self.compute_current_temp = self.compute_current_temp_lin
        elif annealing_type == "exp":
            self.compute_current_temp = self.compute_current_temp_exp

        # Store final rand score and nmi
        self.register_buffer("final_accuracies", torch.zeros(self.hparams.n_clusters))
        self.register_buffer("final_mean_acc", torch.zeros(self.hparams.n_clusters))

        # save num tasks
        self.num_tasks = len(self.hparams.tasks)

    def compute_current_temp_lin(self):
        """
        Compute temperature based on current step
        - linear annealing
        """
        tau_init = self.start_temp
        tau_final = self.final_temp
        rate = (tau_init - tau_final) / float(self.num_steps_annealing)
        tau = max(tau_final, tau_init - rate * self.global_step)
        return tau

    def compute_current_temp_exp(self):
        """
        Compute temperature based on current step
        - exp annealing
        """
        tau_init = self.start_temp
        tau_final = self.final_temp
        rate = -(1 / float(self.num_steps_annealing)) * torch.log(tau_final / tau_init)
        tau = max(tau_final, tau_init * torch.exp(-rate * self.global_step))
        return tau

    def loss_regularizer_n(self, n_hat_onehot):
        ## Create integer number from mvhg sample to compute Pi_Y
        n_hat_integer_cat = torch.cat(n_hat_onehot, dim=1).squeeze(-1).float()
        n_hat_integer_cat = n_hat_integer_cat * torch.arange(
            self.hparams.n_partition_samples + 1, device=self.hparams.device
        ).unsqueeze(0)
        n_hat_integer_cat = n_hat_integer_cat.sum(dim=-1)
        n_hat_integer = torch.cat(
            [
                n_hat_integer_cat[:, i].reshape(-1, 1, 1)
                for i in range(n_hat_integer_cat.shape[-1])
            ],
            dim=-1,
        )
        ## Compute number of possible permutations for current mvhg sample
        log_num_Y_permutations = (
            (self.relu(n_hat_integer) + 1.0).lgamma().squeeze(1).sum(dim=-1)
        )
        loss_reg = log_num_Y_permutations.mean()
        return loss_reg

    def training_step(self, batch, batch_idx, optimizer_idx=1):
        # training_step defines the train loop.
        # it is independent of forward
        images = batch[0]
        labels = {}
        for k, t in enumerate(self.task_keys):
            labels[t] = batch[k + 1]

        # Initialize logger dict
        log = {}

        # Compute current temperature for randpart model
        curr_temp = self.compute_current_temp()
        self.log("temperature", curr_temp)

        # Get current assignment matrix
        assignments, v_logger, rep, out_tasks = self(
            images, g_noise=self.resample, hard=self.hard, tau=curr_temp
        )

        n_hat = torch.cat(v_logger["n_hat"], dim=1)
        log_omega = v_logger["log_omega"]
        log_scores = v_logger["log_scores"]

        loss = torch.zeros(1, device=self.hparams.device)

        losses_tasks = {}
        n_sum = []
        n_hat_mean = n_hat.squeeze(dim=2).mean(dim=0)
        for k, t in enumerate(self.task_keys):
            loss_t = self.hparams.loss_fn[t](out_tasks[t], labels[t])
            losses_tasks[t] = loss_t
            loss += 1/float(self.num_tasks) * loss_t
            # logging task loss
            self.log("loss/train_loss_task_" + t, loss_t, prog_bar=True)
            n_hat_t = n_hat_mean[k]
            self.log("n/n_hat_" + t, n_hat_t, prog_bar=True)
        # loss = loss / len(self.hparams.tasks)
        # Logging loss
        self.log("loss/train_loss_task_sum", loss, prog_bar=True)

        if (
            self.hparams.regularization_partition_index is not None
            or self.hparams.regularization_weight > 0.0
        ):
            loss_reg = self.loss_regularizer_n(v_logger["ohts"])
            loss_reg_weighted = self.hparams.regularization_weight * loss_reg
            loss += loss_reg_weighted
            self.log("loss/train_loss_reg", loss_reg)
            self.log("loss/train_loss_reg_weighted", loss_reg_weighted)

        if n_hat_mean.shape[0] > len(self.hparams.tasks):
            n_hat_not = n_hat_mean[-1]
            self.log("n/n_hat_not", n_hat_not)
        # logging total number of drawn samples
        n_sum = n_hat.sum(dim=[1, 2]).mean()
        self.log("n_sum", n_sum)
        # Logging loss
        self.log("loss/train_loss", loss, prog_bar=True)
        # logging log omega
        log["parameters/log_omega"] = wandb.Histogram(log_omega.detach().cpu())
        log["parameters/log_scores"] = wandb.Histogram(log_scores.detach().cpu())
        self.logger.log_metrics(log)
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
        n_hat_tasks = {}
        sparsity_n_zeros = dict()
        sparsity_n_zeros_neighbourhood = dict()

        for batch, suffix in zip(batches, self.noise_ratio_suffixes):
            images = batch[0]
            labels = {}
            for k, t in enumerate(self.task_keys):
                labels[t] = batch[k + 1]

            assignments, v_logger, rep, inf_tasks = self(images)
            n_hat = torch.cat(v_logger["n_hat"], dim=1)
            n_hat_mean = n_hat.sum(dim=2).mean(dim=0)
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

            # Compute rand score with mean partitioning
            for k, t in enumerate(self.task_keys):
                t_suffix = t + suffix
                inf_t = inf_tasks[t].cpu()
                label_t = labels[t].cpu()
                loss_t = self.hparams.loss_fn[t](inf_t, label_t)
                pred_t = inf_t.data.max(1, keepdim=True)[1]
                acc_t = accuracy_score(label_t, pred_t)
                losses_tasks[t_suffix] = loss_t
                acc_tasks[t_suffix] = acc_t
                n_hat_tasks[t_suffix] = n_hat_mean[k]
                log[f"accuracy/val_accuracy_" + t_suffix] = acc_t
                log["loss/val_loss_" + t_suffix] = loss_t
                log["n/val_n_hat_" + t_suffix] = n_hat_mean[k]

            if n_hat_mean.shape[0] > len(self.hparams.tasks):
                log["n/val_n_hat_not" + suffix] = n_hat_mean[-1]
                n_hat_tasks["not" + suffix] = n_hat_mean[-1]

        # Log validation metrics
        self.logger.log_metrics(log)
        return (
            losses_tasks,
            acc_tasks,
            n_hat_tasks,
            sparsity_n_zeros,
            sparsity_n_zeros_neighbourhood,
        )

    def validation_epoch_end(self, validation_outputs):
        loss_tasks = {}
        acc_tasks = {}
        n_hat_tasks = {}
        has_empty_partition = "not" in validation_outputs[0][2]
        accuracies = []
        for suffix in self.noise_ratio_suffixes:
            for k, t in enumerate(self.task_keys):
                t_suffix = t + suffix
                loss_tasks[t_suffix] = []
                acc_tasks[t_suffix] = []
                n_hat_tasks[t_suffix] = []

            if has_empty_partition:
                n_hat_tasks["not" + suffix] = []

            sparsity_n_zeros = list()
            sparsity_n_zeros_neighbourhood = list()
            for out_batch in enumerate(validation_outputs):
                loss_tasks_batch = out_batch[1][0]
                acc_tasks_batch = out_batch[1][1]
                n_hat_tasks_batch = out_batch[1][2]
                sparsity_n_zeros.append(out_batch[1][3][suffix])
                sparsity_n_zeros_neighbourhood.append(out_batch[1][4][suffix])

                for k, t in enumerate(self.task_keys):
                    t_suffix = t + suffix
                    loss_tasks[t_suffix].append(loss_tasks_batch[t_suffix].item())
                    acc_tasks[t_suffix].append(acc_tasks_batch[t_suffix].item())
                    n_hat_tasks[t_suffix].append(n_hat_tasks_batch[t_suffix].item())

                if has_empty_partition:
                    n_hat_tasks["not" + suffix].append(
                        n_hat_tasks_batch["not" + suffix].item()
                    )

            self.log(
                "sparsity/avg_n_zeros" + suffix, torch.tensor(sparsity_n_zeros).mean()
            )
            self.log(
                "sparsity/avg_n_zeros_neighbourhood" + suffix,
                torch.tensor(sparsity_n_zeros_neighbourhood).mean(),
            )
            for k, t in enumerate(self.task_keys):
                t_suffix = t + suffix
                mean_loss_t = torch.mean(torch.tensor(loss_tasks[t_suffix]))
                mean_acc_t = torch.mean(torch.tensor(acc_tasks[t_suffix]))
                n_hat_t = torch.mean(torch.tensor(n_hat_tasks[t_suffix]))
                self.log("loss/avg_val_loss_" + t_suffix, mean_loss_t)
                self.log("accuracy/avg_acc_" + t_suffix, mean_acc_t)
                self.log("n/avg_n_hat_" + t_suffix, n_hat_t)

                if suffix == self.training_noise_suffix:
                    accuracies.append(mean_acc_t)

            if has_empty_partition:
                n_hat_not = torch.mean(torch.tensor(n_hat_tasks["not" + suffix]))
                self.log("n/avg_n_hat_not" + suffix, n_hat_not)

        mean_task_acc = sum(accuracies) / len(accuracies)
        self.log("accuracy/mean_tasks_acc", mean_task_acc)
        # Save current final scores
        self.final_accuracies = torch.tensor(accuracies)
        self.final_mean_acc = mean_task_acc

    def forward(self, images, tau=None, hard=True, g_noise=False):
        """
        Return a partition matrix that assigns each dimension in x to a cluster
        """
        # Set temperature to be the default if None
        if tau is None:
            tau = self.final_temp

        rep, _ = self.net_rep(images, None)

        # Compute partition
        if self.drpm_use_encoding_model:
            drpm_encoding, _ = self.drpm_encoding_model(images, None)
        else:
            drpm_encoding = rep

        log_score = self.estimate_log_score(drpm_encoding)
        log_omega = self.estimate_log_omega(drpm_encoding)

        assignments, _, ohts, _, n_hat, log_p_n, log_p_pi = self.rpm(
            log_score,
            log_omega,
            g_noise=g_noise,
            hard_n=hard,
            hard_pi=False,
            temperature=tau,
        )
        # compute task based outputs
        out_tasks = {}
        for k, t in enumerate(self.task_keys):
            assign_vec_t = assignments[:, k, :]
            # `assign_vec_t` must have the same number of dimensions as `rep`
            assign_vec_t = assign_vec_t.view(
                (
                    *assign_vec_t.shape,
                    *(1,) * (len(rep.shape) - len(assign_vec_t.shape)),
                )
            )
            rep_t = assign_vec_t * rep
            out_t, _ = self.net_tasks[t](rep_t, None)
            out_tasks[t] = out_t
        v_logger = {
            "log_omega": log_omega,
            "log_scores": log_score,
            "log_p_n": log_p_n,
            "log_p_pi": log_p_pi,
            "n_hat": n_hat,
            "ohts": ohts,
        }
        return assignments, v_logger, rep, out_tasks

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.95, 0.999)
        )
        return {
            "optimizer": optimizer,
        }
