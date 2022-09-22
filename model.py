import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, plm):
        super(Model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method

        if self.cat_method in ['abs', 'dot']:
            linear_in_cnt = 1
        elif self.cat_method in ['cat', 'abs_dot']:
            linear_in_cnt = 2
        elif self.cat_method in ['cat_abs', 'cat_dot']:
            linear_in_cnt = 3
        else:
            linear_in_cnt = 4

        self.MIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        self.SPV_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 3, out_features=args.MODEL.num_classes)
        else:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.SPV_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs):
        # get embeddings from the pretrained language model
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim]

        # H_l ==> H_t for target;  H_r ==> H_b for basic meaning
        tar_mask_ls = (segs_ls == 1).long()
        tar_mask_rs = (segs_rs == 1).long()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)
        H_b = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)

        h_c = torch.mean(embed_l, dim=1)  # context representation
        h_t = torch.mean(H_t, dim=1)  # contextualized target meaning
        h_b = torch.mean(H_b, dim=1)  # basic meaning

        if self.cat_method == 'cat':
            h_mip = torch.cat((h_t, h_b), dim=-1)
            h_spv = torch.cat((h_c, h_t), dim=-1)
        elif self.cat_method == 'abs':
            h_mip = torch.abs(h_t - h_b)
            h_spv = torch.abs(h_c - h_t)
        elif self.cat_method == 'dot':
            h_mip = torch.mul(h_t, h_b)
            h_spv = torch.mul(h_c, h_t)
        elif self.cat_method == 'abs_dot':
            h_mip = torch.cat((torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
            h_spv = torch.cat((torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)
        elif self.cat_method == 'cat_abs':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b)), dim=-1)
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t)), dim=-1)
        elif self.cat_method == 'cat_dot':
            h_mip = torch.cat((h_t, h_b, torch.mul(h_t, h_b)), dim=-1)
            h_spv = torch.cat((h_c, h_t, torch.mul(h_c, h_t)), dim=-1)
        elif self.cat_method == 'cat_abs_dot':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)

        h_mip = self.MIP_linear(h_mip)
        h_spv = self.SPV_linear(h_spv)

        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_mip, h_spv, h_p), dim=-1)
        else:
            final = torch.cat((h_mip, h_spv), dim=-1)

        final = self.dropout3(final)
        out = self.fc(final)  # [batch_size, num_classes]
        return out


class MIP_model(nn.Module):
    def __init__(self, args, plm):
        super(MIP_model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method

        if self.cat_method in ['abs', 'dot']:
            linear_in_cnt = 1
        elif self.cat_method in ['cat', 'abs_dot']:
            linear_in_cnt = 2
        elif self.cat_method in ['cat_abs', 'cat_dot']:
            linear_in_cnt = 3
        else:
            linear_in_cnt = 4

        self.MIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)
        else:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MIP_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs):
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim]

        tar_mask_ls = (segs_ls == 1).long()
        tar_mask_rs = (segs_rs == 1).long()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)
        H_b = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)

        h_t = torch.mean(H_t, dim=1)  # contextualized target meaning
        h_b = torch.mean(H_b, dim=1)  # basic meaning

        if self.cat_method == 'cat':
            h_mip = torch.cat((h_t, h_b), dim=-1)
        elif self.cat_method == 'abs':
            h_mip = torch.abs(h_t - h_b)
        elif self.cat_method == 'dot':
            h_mip = torch.mul(h_t, h_b)
        elif self.cat_method == 'abs_dot':
            h_mip = torch.cat((torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
        elif self.cat_method == 'cat_abs':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b)), dim=-1)
        elif self.cat_method == 'cat_dot':
            h_mip = torch.cat((h_t, h_b, torch.mul(h_t, h_b)), dim=-1)
        elif self.cat_method == 'cat_abs_dot':
            h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)

        h_mip = self.MIP_linear(h_mip)

        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_mip, h_p), dim=-1)
        else:
            final = h_mip

        final = self.dropout3(final)
        out = self.fc(final)  # [batch_size, num_classes]
        return out


class SPV_model(nn.Module):
    def __init__(self, args, plm):
        super(SPV_model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = args.DATA.use_pos
        self.cat_method = args.MODEL.cat_method

        if self.cat_method in ['abs', 'dot']:
            linear_in_cnt = 1
        elif self.cat_method in ['cat', 'abs_dot']:
            linear_in_cnt = 2
        elif self.cat_method in ['cat_abs', 'cat_dot']:
            linear_in_cnt = 3
        else:
            linear_in_cnt = 4

        self.SPV_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        if self.use_pos:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 2, out_features=args.MODEL.num_classes)
        else:
            self.fc = nn.Linear(in_features=args.MODEL.embed_dim * 1, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.SPV_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs):
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim]

        tar_mask_ls = (segs_ls == 1).long()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)

        h_c = torch.mean(embed_l, dim=1)  # context representations
        h_t = torch.mean(H_t, dim=1)  # contextualized target meaning

        if self.cat_method == 'cat':
            h_spv = torch.cat((h_c, h_t), dim=-1)
        elif self.cat_method == 'abs':
            h_spv = torch.abs(h_c - h_t)
        elif self.cat_method == 'dot':
            h_spv = torch.mul(h_c, h_t)
        elif self.cat_method == 'abs_dot':
            h_spv = torch.cat((torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)
        elif self.cat_method == 'cat_abs':
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t)), dim=-1)
        elif self.cat_method == 'cat_dot':
            h_spv = torch.cat((h_c, h_t, torch.mul(h_c, h_t)), dim=-1)
        elif self.cat_method == 'cat_abs_dot':
            h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)

        h_spv = self.SPV_linear(h_spv)

        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((h_spv, h_p), dim=-1)
        else:
            final = h_spv

        final = self.dropout3(final)
        out = self.fc(final)  # [batch_size, num_classes]
        return out


if __name__ == "__main__":
    pass
