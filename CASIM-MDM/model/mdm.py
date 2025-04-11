import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import AutoTokenizer, AutoModel
from model.rotation2xyz import Rotation2xyz


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text']) # [bs, d]
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class CASIM_MDM(MDM):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, save_attn_value=False, clip_final_proj=False, **kargs):
        super().__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout, ablation, activation, legacy, data_rep,
                         dataset, clip_dim, arch, emb_trans_dec, clip_version, **kargs)
        self.save_attn_value = save_attn_value
        self.clip_final_proj = clip_final_proj
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
            # Cross-attention module
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.latent_dim, num_heads=self.num_heads, dropout=self.dropout
            )

        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers,
                                                         norm=nn.LayerNorm(self.latent_dim))
            if self.save_attn_value:
                self.hook_output = SaveOutput()
                self.saved_attention_values = []
                self.saved_attention_masks = []
                patch_attention(self.seqTransDecoder.layers[-1].multihead_attn)
                hook_handle = self.seqTransDecoder.layers[-1].multihead_attn.register_forward_hook(self.hook_output)
            self.causal_mask = None
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
        
    def clear_attention_cache(self):
        self.saved_attention_values = []
        self.saved_attention_masks = []
        self.hook_output.clear()    
    
    def mask_cond(self, cond, force_mask=False):
        bs, d, _ = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        texts = clip.tokenize(raw_text, truncate=True).to(device)

        ## CASIM: token embedding instead of text embedding
        x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # Token embeddings
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # Add positional embeddings

        # Pass through transformer
        x = x.permute(1, 0, 2)  # [batch_size, n_ctx, d_model] -> [n_ctx, batch_size, d_model]
        x = self.clip_model.transformer(x)  # Transformer processes the sequence
        x = x.permute(1, 0, 2)  # [n_ctx, batch_size, d_model] -> [batch_size, n_ctx, d_model]

        # Apply final layer normalization
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        if self.clip_final_proj:
            x = x @ self.clip_model.text_projection

        # Extract the end-of-text ([EOT]) token position
        eot_positions = texts.argmax(dim=-1)  # Position of [EOT] token for each text sequence

        # Compute lengths based on the [EOT] token position
        lengths = eot_positions + 1  # Include the [EOT] token in the length

        # Create a mask based on lengths
        batch_size, n_ctx = texts.shape
        mask = torch.arange(n_ctx, device=texts.device).unsqueeze(0) < lengths.unsqueeze(1)  # [batch_size, n_ctx]

        # Apply the mask element-wise to token embeddings
        x = x * mask.unsqueeze(-1)  # Mask shape expanded to [batch_size, n_ctx, 1] for multiplication

        return x.to(torch.float32), mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get('uncond', False)

        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text, text_mask = y['text_embed']
            else:
                enc_text, text_mask = self.encode_text(y['text']) # [bs, text_len, d]
            
            # mask out certain text embeddings
            emb = emb + self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)).permute(1,0,2) # [text_len, bs, d]
            
        if 'action' in self.cond_mode:
            raise NotImplementedError("CASIM does not support action embedding")

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            T = emb.shape[0]
            xseq = torch.cat((emb, x), axis=0)  # [T+seqlen, bs, d]
            mask = y['mask'][:, 0, 0, :].to(x.device) if 'mask' in y \
                else torch.ones((bs, x.shape[0]), dtype=torch.bool, device=x.device) # B, seqlen
            src_key_padding_mask = ~torch.cat((text_mask, mask), axis=1) # B, T + seqlen
            xseq = self.sequence_pos_encoder(xseq)  # [T+seqlen, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)[T:]  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]

            if self.save_attn_value and force_mask==False:
                mask = ~y['mask'][:, 0, 0, :].to(x.device) if 'mask' in y else None # B, seqlen
                output = self.seqTransDecoder(tgt=xseq, 
                                              tgt_key_padding_mask=mask,
                                              memory=emb,
                                              memory_key_padding_mask=~text_mask,)
                self.saved_attention_values.append(self.hook_output.outputs[-1])
                self.saved_attention_masks.append(text_mask)
                
            else:
                mask = ~y['mask'][:, 0, 0, :].to(x.device) if 'mask' in y else None # B, seqlen
                output = self.seqTransDecoder(tgt=xseq, 
                                              tgt_key_padding_mask=mask,
                                              memory=emb,
                                              memory_key_padding_mask=~text_mask,)
            
        elif self.arch == 'gru':
            raise NotImplementedError("CASIM does not support gru for now")

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output
        
    def apply_cross_attention(self, x, enc_text_emb, text_mask):
        """
        Apply cross-attention between x and enc_text_emb.
        """
        attn_output, _ = self.cross_attention(
            query=x,  # Query: input sequence [seq_len, batch_size, latent_dim]
            key=enc_text_emb,  # Key: text embedding [text_len, batch_size, latent_dim]
            value=enc_text_emb,  # Value: text embedding [text_len, batch_size, latent_dim]
            key_padding_mask=~text_mask  # Mask for text tokens (optional) [batch_size, text_len]
        )
        return attn_output



class CASIM_MDM_Bert(MDM):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout, ablation, activation, legacy, data_rep,
                         dataset, clip_dim, arch, emb_trans_dec, clip_version, **kargs)
        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
            # Cross-attention module
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.latent_dim, num_heads=self.num_heads, dropout=self.dropout
            )

        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers,
                                                         norm=nn.LayerNorm(self.latent_dim))
            self.causal_mask = None
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        bert_model_name='bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_model.eval()
        for param in self.bert_model.parameters():
            param.requires_grad = False  # Freeze BERT weights
        self.embed_text = nn.Linear(768, self.latent_dim)
        del self.clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d, _ = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    @torch.no_grad()
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device

        # Tokenize input text
        encoding = self.tokenizer(raw_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoding['input_ids'].to(device)  # Token IDs
        attention_mask = encoding['attention_mask'].to(device)  # Attention mask

        # Pass through BERT model
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

        return token_embeddings, attention_mask.to(torch.bool)

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get('uncond', False)

        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text, text_mask = y['text_embed']
            else:
                enc_text, text_mask = self.encode_text(y['text']) # [bs, text_len, d]
            
            # mask out certain text embeddings
            emb = emb + self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)).permute(1,0,2) # [text_len, bs, d]
            
        if 'action' in self.cond_mode:
            raise NotImplementedError("CASIM does not support action embedding")

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            T = emb.shape[0]
            xseq = torch.cat((emb, x), axis=0)  # [T+seqlen, bs, d]
            mask = y['mask'][:, 0, 0, :].to(x.device) if 'mask' in y \
                else torch.ones((bs, x.shape[0]), dtype=torch.bool, device=x.device) # B, seqlen
            src_key_padding_mask = ~torch.cat((text_mask, mask), axis=1) # B, T + seqlen
            xseq = self.sequence_pos_encoder(xseq)  # [T+seqlen, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)[T:]  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            mask = ~y['mask'][:, 0, 0, :].to(x.device) if 'mask' in y else None # B, seqlen

            output = self.seqTransDecoder(tgt=xseq, 
                                          tgt_key_padding_mask=mask,
                                          memory=emb,
                                          memory_key_padding_mask=~text_mask,) # may need to add causal mask and tgt_key_padding_mask
        elif self.arch == 'gru':
            raise NotImplementedError("CASIM does not support gru for now")

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output
        
    def apply_cross_attention(self, x, enc_text_emb, text_mask):
        """
        Apply cross-attention between x and enc_text_emb.
        """
        attn_output, _ = self.cross_attention(
            query=x,  # Query: input sequence [seq_len, batch_size, latent_dim]
            key=enc_text_emb,  # Key: text embedding [text_len, batch_size, latent_dim]
            value=enc_text_emb,  # Value: text embedding [text_len, batch_size, latent_dim]
            key_padding_mask=~text_mask  # Mask for text tokens (optional) [batch_size, text_len]
        )
        return attn_output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    
class SaveOutput: # Save the hooked output
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []
        
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap
    
import math
class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head
    def forward(self, x, encoder_mask=None):
        B, T, C = x.size() 
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        if encoder_mask is not None:
            T_t = encoder_mask.shape[1]
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.n_head, T, 1)
            att[..., :T, :T_t] = att[..., :T, :T_t].masked_fill(encoder_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y