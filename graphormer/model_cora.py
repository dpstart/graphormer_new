# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from utils.flag import flag_bounded


def init_bert_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Graphormer(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_virtual_tokens = 1
        self.num_heads = num_heads

        self.atom_encoder = nn.Linear(1433, hidden_dim)
        self.edge_encoder = nn.Linear(1, num_heads)
        self.edge_type = edge_type
        self.rel_pos_encoder = nn.Linear(1, num_heads)
        self.in_degree_encoder = nn.Linear(1, hidden_dim)
        self.out_degree_encoder = nn.Linear(1, hidden_dim)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.downstream_out_proj = nn.Linear(
            hidden_dim, get_dataset(dataset_name)["num_class"]
        )

        self.graph_token = nn.Embedding(num_virtual_tokens, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(num_virtual_tokens, num_heads)

        self.evaluator = get_dataset(dataset_name)["evaluator"]
        self.metric = get_dataset(dataset_name)["metric"]
        self.loss_fn = get_dataset(dataset_name)["loss_fn"]
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_bert_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):

        num_virtual_tokens = 1
        attn_bias, rel_pos, x = (
            batched_data.attn_bias,
            batched_data.rel_pos,
            batched_data.x,
        )
        in_degree, out_degree = (
            batched_data.in_degree.float(),
            batched_data.in_degree.float(),
        )
        edge_input, attn_edge_type = (
            batched_data.edge_input.float(),
            batched_data.attn_edge_type.float(),
        )
        adj = batched_data.adj

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_pos_bias = self.rel_pos_encoder(rel_pos.float().unsqueeze(-1)).permute(
            0, 3, 1, 2
        )
        graph_attn_bias[:, :, num_virtual_tokens:, num_virtual_tokens:] = (
            graph_attn_bias[:, :, num_virtual_tokens:, num_virtual_tokens:]
            + rel_pos_bias
        )
        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(
            1, self.num_heads, num_virtual_tokens
        ).unsqueeze(-2)
        self.graph_token_virtual_distance.weight.view(
            1, self.num_heads, num_virtual_tokens
        ).unsqueeze(-2)
        graph_attn_bias[:, :, num_virtual_tokens:, :num_virtual_tokens] = (
            graph_attn_bias[:, :, num_virtual_tokens:, :num_virtual_tokens] + t
        )

        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.edge_encoder(attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, :, :] = graph_attn_bias[:, :, :, :] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree.T)
            + self.out_degree_encoder(out_degree.T)
        )
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature.unsqueeze(0)], dim=1
        )

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(
                output, graph_attn_bias, mask=None
            )  # TODO readd mask as adj
        output = self.final_ln(output)
        output = self.downstream_out_proj(output)
        return output

    def training_step(self, batched_data, batch_idx):

        y_hat = self(batched_data)
        y_gt = batched_data.y.view(-1)
        train_mask = get_dataset(self.dataset_name)["dataset"].data.train_mask
        loss = self.loss_fn(y_hat[0, 1:-3, :][train_mask], y_gt[train_mask])
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):

        y_hat = self(batched_data)
        y_gt = batched_data.y.view(-1)
        val_mask = get_dataset(self.dataset_name)["dataset"].data.val_mask
        loss = self.loss_fn(y_hat[0, 1:-3, :][val_mask], y_gt[val_mask])
        return {
            "y_pred": y_hat[0, 1:-3, :][val_mask],
            "y_true": y_gt[val_mask],
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i["y_pred"] for i in outputs])
        y_true = torch.cat([i["y_true"] for i in outputs])

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        try:
            self.log(
                "valid_" + self.metric, self.loss_fn(y_pred, y_true), sync_dist=True,
            )
        except:
            pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ["PCQM4M-LSC", "ZINC"]:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "idx": batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i["y_pred"] for i in outputs])
        y_true = torch.cat([i["y_true"] for i in outputs])
        if self.dataset_name == "PCQM4M-LSC":
            result = y_pred.cpu().float().numpy()
            idx = torch.cat([i["idx"] for i in outputs])
            torch.save(result, "y_pred.pt")
            torch.save(idx, "idx.pt")
            exit(0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log(
            "test_" + self.metric,
            self.evaluator.eval(input_dict)[self.metric],
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument("--n_layers", type=int, default=12)
        parser.add_argument("--num_heads", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--ffn_dim", type=int, default=512)
        parser.add_argument("--intput_dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--attention_dropout_rate", type=float, default=0.1)
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--warmup_updates", type=int, default=60000)
        parser.add_argument("--tot_updates", type=int, default=1000000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument("--edge_type", type=str, default="multi_hop")
        parser.add_argument("--validate", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument("--flag", action="store_true")
        parser.add_argument("--flag_m", type=int, default=3)
        parser.add_argument("--flag_step_size", type=float, default=1e-3)
        parser.add_argument("--flag_mag", type=float, default=1e-3)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, mask=mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
