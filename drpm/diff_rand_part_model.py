import torch
import torch.nn as nn
import torch.nn.functional as F


from mvhg.pt_fmvhg import MVHG
from drpm.pl import PL


class RandPart(nn.Module):
    def __init__(
        self, n_cluster, tau_mult=1.0, perm_log_p_pi_n=True, eps=1e-10, device="cuda"
    ):
        super().__init__()
        self.n_cluster = n_cluster
        self.device = device

        # Instantiate MVHG layer
        self.mvhg = MVHG(device=self.device)

        # Empty initialization of precomputed shift matrices
        # Compute lazily as its batch size dependent
        self.shift_matrix_selector = torch.empty(0, 0)

        # multiplication factor of temperature for softmax
        # compared to softmax temperature of mvhg
        self.tau_mult = tau_mult
        self.eps = eps

        # boolean to define calculation of log p(pi | n)
        self.perm_log_p_pi_n = perm_log_p_pi_n

    def precompute_shift_matrix_selector(self, dim):
        """
        Precompute helper tensor that allows the computation of a matrix that shifts vector entries by simple multiplication operations
        """

        with torch.no_grad():
            self.shift_matrix_selector = torch.zeros(
                dim + 1, dim, dim, dtype=torch.float32, device=self.device
            )
            for i in range(dim + 1):
                matrix = torch.diag_embed(torch.ones(dim - i), offset=-i)
                self.shift_matrix_selector[i] = matrix.type_as(
                    self.shift_matrix_selector
                )
            self.shift_matrix_selector = self.shift_matrix_selector.unsqueeze(0)

    def sample_mvhg(self, m, n, w, g_noise=True, hard=True, temperature=1e-2):
        """
        Sample from the multivariate hyper geometric distribution with |m| colors and m_i balls per color with n draws and color weights w_i
        """
        return self.mvhg(
            m,
            n,
            w,
            temperature=temperature,
            g_noise=g_noise,
            hard=hard,
        )

    def shift_filled_ohts(self, filled_ohts):
        """
        Shift filled one hot encoders so that they add up to a one vector
        """
        shifted_filled_ohts = torch.zeros_like(filled_ohts)
        shift_selector_vec = torch.zeros(
            filled_ohts.shape[0], filled_ohts.shape[1] + 1, 1, 1
        ).type_as(filled_ohts)
        for i in range(self.n_cluster):
            selector = -(
                shift_selector_vec
                - torch.cat(
                    [
                        torch.ones_like(shift_selector_vec[:, 0]).unsqueeze(1),
                        shift_selector_vec[:, :-1],
                    ],
                    dim=1,
                )
            )
            # Select shift matrix
            shift_mat = (selector * self.shift_matrix_selector).sum(dim=1)
            # Shift hyper geometric output to correct position
            shifted_filled_ohts[:, :, i] = torch.bmm(
                shift_mat, filled_ohts[:, :, i].unsqueeze(-1)
            ).squeeze(-1)

            # Update shift selector vector
            shift_selector_vec[:, :-1] = shift_selector_vec[
                :, :-1
            ] + shifted_filled_ohts[:, :, i].unsqueeze(-1).unsqueeze(-1)
        return shifted_filled_ohts.permute(0, 2, 1)

    def get_partition_sizes(
        self, num_samples, log_omega, g_noise=True, hard=True, temperature=1e-2
    ):
        """
        Return two vectors shifted_filled_ohts and num_per_cluster
        shifted_filled_ohts: matrix of shape batch_size x num_samples x n_clusters containing partition sizes encoded as partial one vectors that can be useful to collaps permutation matrices to multi oht vectors assigning elements to clusters
        num_per_cluster: vector of size batch_size x n_cluster containing the number of elements per cluster
        """
        # Get batch_size
        batch_size = log_omega.shape[0]

        # Precompute filled oht vector  and shift matrix selector if necessary
        if self.shift_matrix_selector.shape[-1] != num_samples:
            self.precompute_shift_matrix_selector(num_samples)

        # For each element in batch, sample num_samples times from urn model with log_omega parameter for ball color weights
        m = torch.tensor([num_samples for i in range(self.n_cluster)]).type_as(
            log_omega
        )
        n = (
            torch.tensor(num_samples)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .type_as(log_omega)
        )
        ohts, num_per_cluster, filled_ohts, log_p = self.mvhg(
            m, n, log_omega, temperature, add_noise=g_noise, hard=hard
        )

        cat_filled_ohts = torch.cat(filled_ohts, dim=1).permute(0, 2, 1)[:, 1:, :]

        # Shift filled ohts to correct location in order to collapse permutation matrix
        shifted_filled_ohts = self.shift_filled_ohts(cat_filled_ohts)

        return ohts, shifted_filled_ohts, num_per_cluster, log_p

    def get_partition(self, log_scores, filled_ohts, temperature, add_noise, hard):
        # Sort score vector with neuralsort
        sort = PL(
            log_scores, tau=self.tau_mult * temperature, g_noise=add_noise, hard=hard
        )
        permutation = sort.rsample([1])

        # Calculate random partition
        partition = filled_ohts @ permutation
        partition = torch.clip(partition, 0.0, 1.0)
        # calculate log probability log(p(Y | n))
        if self.perm_log_p_pi_n:
            log_p_pi_n = sort.log_prob(permutation)
        else:
            log_p_pi_n = self.get_log_prob_neuralsort(
                log_scores, partition, filled_ohts
            )
        return permutation, partition, log_p_pi_n

    def get_log_prob_mvhg(
        self,
        log_omega,
        x_hat,
        y_hat,
    ):
        # Get batch_size and number of samples
        batch_size = log_omega.shape[0]
        if torch.is_tensor(x_hat):
            num_samples = x_hat[0].sum().clone().detach()
        else:
            num_samples = torch.cat(x_hat, dim=1)[0].sum().clone().detach()

        # For each element in batch, sample num_samples times from urn model with log_omega parameter for ball color weights
        m = torch.tensor([num_samples for i in range(self.n_cluster)]).type_as(
            log_omega
        )
        n = num_samples.unsqueeze(0).repeat(batch_size, 1).type_as(log_omega)
        log_p = self.mvhg.get_log_probs(m, n, x_hat, y_hat, log_omega)
        return log_p

    def get_log_prob_permutation(
        self,
        log_scores: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        # Make Plackett-Luce distribution of log_scores and find probability of permutation
        sort = PL(log_scores, tau=None)
        log_p = sort.log_prob(permutation)
        return log_p

    def compute_conditioned_log_scores(
        self,
        sort_log_score,
        conditional_log_score_selection,
        shifted_filled_ohts,
        temperature=1.0,
    ):
        # Get permutation based on log_scores
        sort = PL(
            sort_log_score, tau=self.tau_mult * temperature, g_noise=False, hard=True
        )
        permutation = sort.rsample([1])

        # Use permutation to assign log_scores for partition
        log_score_selection = conditional_log_score_selection.unsqueeze(-1).repeat(
            1, 1, shifted_filled_ohts.shape[-1]
        )
        log_score_selection = (log_score_selection * shifted_filled_ohts).sum(
            dim=1, keepdim=True
        )
        log_scores = (log_score_selection @ permutation).squeeze(1)

        return log_scores

    def get_log_prob_neuralsort(
        self,
        log_scores: torch.Tensor,
        partition: torch.Tensor,
        filled_ohts: torch.Tensor,
    ) -> torch.Tensor:
        # approximate partition probability using condensed form of permutation
        # matrix
        sort = PL(log_scores, tau=1.0, g_noise=False, hard=False)
        permutation = sort.relaxed_sort(log_scores.unsqueeze(-1), hard=False)
        sum_probs = filled_ohts @ permutation
        sum_probs_f = partition * sum_probs
        sum_probs_ff = sum_probs_f.sum(dim=1)+1e-10
        log_sum_probs = sum_probs_ff.log()
        log_p_batch = log_sum_probs.sum(dim=1)
        log_p = log_p_batch.mean()
        return log_p

    def get_log_prob(
        self,
        partition: torch.Tensor,
        log_omega: torch.Tensor,
        log_scores: torch.Tensor,
        filled_ohts: torch.Tensor,
        x_hat: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> torch.Tensor:
        log_prob_mvhg = self.get_log_prob_mvhg(log_omega, x_hat, y_hat)
        log_prob_pi_n = self.get_log_prob_neuralsort(log_scores, partition, filled_ohts)
        log_prob_part = log_prob_mvhg + log_prob_y_n
        return log_prob_part, log_prob_mvhg, log_prob_y_n

    def forward(
        self,
        sort_log_score,
        mvhg_log_omega,
        g_noise=True,
        hard_n=True,
        hard_pi=True,
        temperature=1e-2,
        conditional_log_score_selection=None,
    ):
        assert (
            mvhg_log_omega.shape[-1] == self.n_cluster
        ), f"Provided omega has {mvhg_log_omega.shape[-1]} cluster weights but expected {self.n_cluster}."
        # Get batch size and number of samples in set to partition
        batch_size, num_samples = sort_log_score.shape

        # Generate partition sizes with mvhg
        (
            ohts,
            shifted_filled_ohts,
            num_per_cluster,
            log_p_mvhg,
        ) = self.get_partition_sizes(
            num_samples,
            mvhg_log_omega,
            g_noise=g_noise,
            hard=hard_n,
            temperature=temperature,
        )
        if conditional_log_score_selection is not None:
            sort_log_score = self.compute_conditioned_log_scores(
                sort_log_score,
                conditional_log_score_selection,
                shifted_filled_ohts,
                temperature=temperature,
            )
        perm, part, log_p_pi_n = self.get_partition(
            sort_log_score, shifted_filled_ohts, temperature, g_noise, hard_pi
        )
        log_p_rpm = log_p_mvhg + log_p_pi_n
        if conditional_log_score_selection is None:
            return (
                part,
                perm,
                ohts,
                shifted_filled_ohts,
                num_per_cluster,
                log_p_mvhg,
                log_p_pi_n,
            )
        else:
            return (
                part,
                perm,
                ohts,
                shifted_filled_ohts,
                num_per_cluster,
                log_p_mvhg,
                log_p_pi_n,
                sort_log_score,
            )
