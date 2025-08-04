import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg
import numpy as np

class AdaptiveIncrementalAttention(nn.Module):
    def __init__(self, embedding_dim, region_dim):
        super(AdaptiveIncrementalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.region_dim = region_dim

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_e = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.omega = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, e_i_t_minus_1, e_j_t, imp_j_t, R_lk):
        e_i_expanded = e_i_t_minus_1.unsqueeze(1).expand_as(e_j_t)
        q_i = self.W_q(e_i_expanded)
        k_j = self.W_k(e_j_t)
        # (batch_size, num_events_j, 1)
        sigma_ij = torch.tanh(q_i + k_j).matmul(self.omega).unsqueeze(2)

        exp_sigma = torch.exp(sigma_ij)
        sum_exp_sigma = torch.sum(exp_sigma, dim=1, keepdim=True)
        a_ij = (imp_j_t.unsqueeze(2) * exp_sigma) / (sum_exp_sigma + 1e-9)

        v_j = self.W_v(e_j_t)
        A_i_t_minus_1_t = torch.sum(a_ij * v_j, dim=1)

        # (batch_size, num_events_j, embedding_dim)
        Wv_ejt = self.W_v(e_j_t)
        # (batch_size, num_events_j, 1, embedding_dim) -> (batch_size, num_events_j, region_dim, embedding_dim)
        term_to_sum_lk = R_lk.unsqueeze(0).unsqueeze(0) * (a_ij.unsqueeze(3) * Wv_ejt.unsqueeze(2))
        # (batch_size, region_dim, embedding_dim)
        A_i_t_minus_1_t_lk = torch.sum(term_to_sum_lk, dim=1)
        A_i_t_minus_1_t_l = torch.sum(a_ij * self.W_v(e_j_t), dim=1)

        return A_i_t_minus_1_t, A_i_t_minus_1_t_lk, A_i_t_minus_1_t_l


def update_accumulative_semantic(A_i_t_minus_1_t_l, A_i_t_minus_1_t_lk, N_c, K):
    A_c_t_minus_1_t_l = (1 / N_c) * torch.sum(A_i_t_minus_1_t_l, dim=0)
    sum_over_i = torch.sum(A_i_t_minus_1_t_lk, dim=0)
    sum_over_k = torch.sum(sum_over_i, dim=0)
    A_c_t_minus_1_t_l_bar = (1 / K) * (1 / N_c) * sum_over_k

    return A_c_t_minus_1_t_l


class MultiRegionalFusionModule(nn.Module):
    def __init__(self, total_countries: int, regional_countries_in_context: int,
                 regional_indices_in_correlations: torch.Tensor = None, device='cpu'):
        super().__init__()
        self.N_total_countries = total_countries
        self.N_regional_countries_in_context = regional_countries_in_context
        self.device = device 
        self.region_correlations = nn.Parameter(torch.randn(total_countries, total_countries))

        if regional_indices_in_correlations is None:
            regional_indices_in_correlations = torch.arange(regional_countries_in_context, dtype=torch.long)

        self.register_buffer('regional_indices_for_corr_cols', regional_indices_in_correlations.to(device))

        assert self.regional_indices_for_corr_cols.shape[0] == regional_countries_in_context, \
            "Shape of regional_indices_in_correlations must match N_regional_countries_in_context"

    def forward(self,
                target_country_vectors: torch.Tensor,  # (B, n_target, s_h, F_h)
                regional_vectors: torch.Tensor,  # (B, N_regional, F_h, s_h) <- Note: S_r_prime=F_h, F_r_prime=s_h
                target_country_indices: torch.Tensor):  # (B, n_target), dtype=torch.long

        b, n_target, s_h, f_h = target_country_vectors.shape
        _B_reg, n_regional_from_input, s_r_prime, f_r_prime = regional_vectors.shape

        assert b == _B_reg, "Batch sizes of target and regional vectors must match."
        assert n_regional_from_input == self.N_regional_countries_in_context, \
            "Input N_regional must match N_regional_countries_in_context set during init."
        assert s_h == f_r_prime and f_h == s_r_prime, \
            f"Dimension mismatch for fusion: target (s_h:{s_h}, F_h:{f_h}), regional_input (F_h:{s_r_prime}, s_h:{f_r_prime})"

        target_country_indices = target_country_indices.to(self.region_correlations.device)
        correlations_for_targets_all_cols = self.region_correlations[target_country_indices, :]
        correlations_for_region = correlations_for_targets_all_cols[:, :, self.regional_indices_for_corr_cols]
        attention_weights = F.softmax(correlations_for_region, dim=2)  # (B, n_target, N_regional)
        context_permuted_to_match_target_dims = torch.einsum('bkji,btk->btji', regional_vectors, attention_weights)
        aligned_context = context_permuted_to_match_target_dims.permute(0, 1, 3, 2)
        updated_target_vectors = target_country_vectors + aligned_context

        return updated_target_vectors

class AEGIS(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        self.config = config
        self.device = task.device
        self.threshold = config.threshold
        self.dropout = config.dropout
        self.feas_names = task.main_feas
        self.horizon = config.horizon
        self.num_countries = task.num_countries
        self.country_nodes = task.country_nodes
        self.event_embedding_size = config.event_fea_dim
        self.num_events = config.num_events
        self.bs_idx = task.bs2idx
        self.r_idx = task.r2idx
        self.norm_epsilon = 1e-9
        self.context_weight = config.context_weight
        self.attn_net = AdaptiveIncrementalAttention(self.event_embedding_size, self.event_embedding_size)

        self.events = torch.cat([task.train_events, task.val_events, task.test_events], dim=0).to(self.device)
        self.event_type_embedding = nn.Embedding(num_embeddings=self.num_events, embedding_dim=self.event_embedding_size)

        self.region_fusion_module = MultiRegionalFusionModule(
            total_countries=self.num_countries,
            regional_countries_in_context=self.num_countries,
            device=self.device,
        )

        self.event_semantic_gru_layer = nn.GRU(input_size=self.event_embedding_size,
                                               hidden_size=self.event_embedding_size,
                                               num_layers=2,
                                               batch_first=True,
                                               dropout=self.dropout)

        self.final_dropout = nn.Dropout(self.dropout)
        self.final_classifier = SimpleBinaryClassifier(input_features=self.event_embedding_size, hidden_size1=self.event_embedding_size * 2,
                                                       hidden_size2=self.event_embedding_size, dropout_rate=self.dropout)

        self.apply(_xavier_init_weights)

    def forward(self, inputs, event_ids, date_ids, country_ids):
        b = event_ids.size(0)
        assert inputs.shape[0] == event_ids.size(0)

        batch_events_tensors = self.events[date_ids, country_ids, self.inputs[0], self.inputs[1]]

        A_i_t, A_i_lk, A_i_l = self.attn_net(batch_events_tensor[:-1], batch_events_tensor)
        A_c_l = update_accumulative_semantic(all_A_i_l, all_A_i_lk, N_c, region_dim)
        batch_events_tensor = torch.matmul(A_c_l)

        all_event_embeddings = self.event_type_embedding.weight
        current_event_semantic_embeddings = all_event_embeddings[event_ids]

        bs_idx_tensor = torch.tensor(self.bs_idx).to(self.device)
        r_idx_tensor = torch.tensor(self.r_idx).to(self.device)

        bs_mask = torch.isin(event_ids, bs_idx_tensor)
        r_mask = torch.isin(event_ids, r_idx_tensor)

        country_semantic_tensor = torch.zeros(b, self.event_embedding_size, self.horizon, device=self.device, dtype=all_event_embeddings.dtype)

        num_bs = bs_mask.sum()
        if num_bs > 0:
            bs_event_semantic = current_event_semantic_embeddings[bs_mask] # (num_bs, EES)
            original_batch_indices = torch.arange(b, device=self.device)
            bs_date_ids = date_ids[bs_mask]
            bs_country_ids = country_ids[bs_mask]
            bs_original_indices_for_context = original_batch_indices[bs_mask]

            normed_tensors_bs = self.events[bs_date_ids, bs_country_ids, bs_original_indices_for_context, :]
            outer_products_bs = torch.bmm(bs_event_semantic.unsqueeze(2), normed_tensors_bs.unsqueeze(1))   # (num_bs, EES, D_feat_raw)
            country_semantic_tensor[bs_mask] = outer_products_bs

        num_r = r_mask.sum()
        if num_r > 0:
            r_context_events = batch_events_tensors[r_mask]  # Shape: [num_r, num_events, horizon]
            sum_tensor_r = torch.sum(r_context_events, dim=1, keepdim=True)  # Shape: [num_r, 1, horizon]
            normed_tensors_r = r_context_events / (sum_tensor_r + self.norm_epsilon)  # Shape: [num_r, num_events, horizon]
            einsum_results_r = torch.einsum('vs,nvd->nsd', all_event_embeddings, normed_tensors_r)
            country_semantic_tensor[r_mask] = einsum_results_r

        base_primary_country_rep = country_semantic_tensor.permute(0, 2, 1)

        initial_target_vectors = base_primary_country_rep.unsqueeze(1).expand(-1, self.num_countries, -1, -1)
        region_event_tensor_all_countries = self.events[date_ids].permute(0, 1, 3, 2)

        sum_val = region_event_tensor_all_countries.sum(dim=3, keepdim=True)
        normed_regional_events = region_event_tensor_all_countries / (sum_val + self.norm_epsilon)

        regional_vectors4fusion = torch.einsum('bnhv,ve->bnhe', normed_regional_events, all_event_embeddings).permute(0, 1, 3, 2)
        target_indices_all = torch.arange(self.num_countries, device=self.device).long().unsqueeze(0).expand(b, -1)
        updated_all_country_vectors = self.region_fusion_module(initial_target_vectors, regional_vectors4fusion, target_indices_all)

        fused_primary_country_vector = updated_all_country_vectors[torch.arange(b, device=self.device), country_ids, :,:]
        final_vector_for_gru = (1.0 - self.context_weight) * base_primary_country_rep + \
                               self.context_weight * fused_primary_country_vector

        lstm_o, _ = self.event_semantic_gru_layer(final_vector_for_gru)
        last_time_step_output = lstm_o[:,-1,:].squeeze()
        logits = self.final_classifier(last_time_step_output).squeeze()

        return logits, event_ids

class NaiveGRU(nn.Module):
    def __init__(self, config, task):
        super(NaiveGRU, self).__init__()
        self.config = config
        self.device = task.device
        self.threshold = config.threshold
        self.dropout = config.dropout
        self.event_embedding_size = config.event_fea_dim
        self.num_countries = task.num_countries

        self.gru = nn.GRU(input_size=self.num_countries + 2,
                          hidden_size=self.event_embedding_size,
                          num_layers=2,
                          batch_first=True,
                          dropout=self.dropout)

        self.classifier = SimpleBinaryClassifier(input_features=self.event_embedding_size, hidden_size1=self.event_embedding_size * 2,
                                                 hidden_size2=self.event_embedding_size, dropout_rate=self.dropout)

    def forward(self, inputs, event_ids, date_ids, country_ids):
        b = event_ids.size(0)
        assert inputs.shape[0] == event_ids.size(0)

        inputs = inputs.squeeze()
        o, _ = self.gru(inputs)
        last_time_step_output = o[:, -1, :].squeeze()
        logits = self.classifier(last_time_step_output).squeeze()

        return logits, event_ids

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_features, hidden_size1, hidden_size2=None, dropout_rate=0.2):
        super(SimpleBinaryClassifier, self).__init__()

        self.layer_1 = nn.Linear(input_features, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        if hidden_size2:
            self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.output_layer = nn.Linear(hidden_size2, 1)
        else:
            self.layer_2 = None
            self.output_layer = nn.Linear(hidden_size1, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        if self.layer_2:
            x = self.layer_2(x)
            x = self.relu2(x)
            x = self.dropout2(x)

        logits = self.output_layer(x)
        return logits

def _xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
    elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('sigmoid'))
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('tanh'))
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if m.data.ndim > 1:
            nn.init.xavier_uniform_(m.data, gain=1.0)
        else:
            pass