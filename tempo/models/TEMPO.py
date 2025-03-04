import os
import warnings

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from einops import rearrange
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import GPT2Tokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from tempo.utils.rev_in import RevIn
from tempo.utils.tools import sample_negative_binomial

from .MovingAverage import MovingAverage
from .MultiFourier import MultiFourier


class TEMPO(nn.Module):
    """
    TODO: create docstring for TEMPO class
    """

    def __init__(self, configs):
        super(TEMPO, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.mul_season = MultiFourier([2], [24 * 4])
        self.seq_len = configs.seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.distr_output = StudentTOutput()
        self.hidden_dim = configs.d_model
        # self.mlp = configs.mlp
        self.criterion = nn.MSELoss()
        self.map_trend = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_season = nn.Sequential(
            nn.Linear(configs.seq_len, 4 * configs.seq_len),
            nn.ReLU(),
            nn.Linear(4 * configs.seq_len, configs.seq_len),
        )
        # #self.map_season = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_resid = nn.Linear(configs.seq_len, configs.seq_len)
        self.moving_average = MovingAverage(kernel_size=25, stride=1)

        if configs.is_gpt:
            if configs.pretrain:
                # load a pre-trained GPT-2 base model
                self.gpt2_trend = GPT2Model.from_pretrained(
                    "gpt2",
                    output_attentions=True,
                    output_hidden_states=True,
                )
                # self.gpt2_season = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                # self.gpt2_noise = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("-" * 18 + "No need to load pretrained GPT model" + "-" * 18)
                self.gpt2_trend = GPT2Model(GPT2Config())
                # self.gpt2_season = GPT2Model(GPT2Config())
                # self.gpt2_noise = GPT2Model(GPT2Config())
            self.gpt2_trend.h = self.gpt2_trend.h[: configs.gpt_layers]
            self.prompt = configs.prompt
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_trend_token = self.tokenizer(
                text="Predict the future time step given the trend",
                return_tensors="pt",
            )
            self.gpt2_season_token = self.tokenizer(
                text="Predict the future time step given the season",
                return_tensors="pt",
            )
            self.gpt2_residual_token = self.tokenizer(
                text="Predict the future time step given the residual",
                return_tensors="pt",
            )

            self.token_len = len(self.gpt2_trend_token["input_ids"][0])

            try:
                self.pool = configs.pool
                if self.pool:
                    self.prompt_record_plot = {}
                    self.prompt_record_id = 0
                    self.diversify = True
            except Exception as _:
                self.pool = False

            if self.pool:
                self.prompt_key_dict = nn.ParameterDict({})
                self.prompt_value_dict = nn.ParameterDict({})
                # self.summary_map = nn.Linear(self.token_len, 1)
                self.summary_map = nn.Linear(self.patch_num, 1)
                self.pool_size = 30
                self.top_k = 3
                self.prompt_len = 3
                self.token_len = self.prompt_len * self.top_k
                for i in range(self.pool_size):
                    prompt_shape = (self.prompt_len, 768)
                    key_shape = 768
                    self.prompt_value_dict[f"prompt_value_{i}"] = nn.Parameter(
                        torch.randn(prompt_shape)
                    )
                    self.prompt_key_dict[f"prompt_key_{i}"] = nn.Parameter(
                        torch.randn(key_shape)
                    )

                self.prompt_record = {f"id_{i}": 0 for i in range(self.pool_size)}
                self.prompt_record_trend = {}
                self.prompt_record_season = {}
                self.prompt_record_residual = {}
                self.diversify = True

        self.in_layer_trend = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_season = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_noise = nn.Linear(configs.patch_size, configs.d_model)
        # self.out_layer_noise = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.prompt == 1:
            self.use_token = configs.use_token

            # if use prompt token's representation as the forecasting's information
            if self.use_token == 1:
                in_features = configs.d_model * (self.patch_num + self.token_len)
                out_features = configs.pred_len
            else:
                in_features = configs.d_model * self.patch_num
                out_features = configs.pred_len

            self.out_layer_trend = nn.Linear(in_features, out_features)
            self.out_layer_season = nn.Linear(in_features, out_features)
            self.out_layer_noise = nn.Linear(in_features, out_features)
            # self.fre_len = configs.seq_len # // 2 + 1
            # self.out_layer_noise_fre = ComplexLinear(self.fre_len, configs.pred_len)
            # self.pred_len = configs.pred_len
            # self.seq_len = configs.seq_len

            self.prompt_layer_trend = nn.Linear(configs.d_model, configs.d_model)
            self.prompt_layer_season = nn.Linear(configs.d_model, configs.d_model)
            self.prompt_layer_noise = nn.Linear(configs.d_model, configs.d_model)
        else:
            in_features = configs.d_model * self.patch_num
            out_features = configs.pred_len
            self.out_layer_trend = nn.Linear(in_features, out_features)
            self.out_layer_season = nn.Linear(in_features, out_features)
            self.out_layer_noise = nn.Linear(in_features, out_features)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2_trend.named_parameters()):
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=16,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",  # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )

        self.gpt2_trend = get_peft_model(self.gpt2_trend, config)
        self.print_trainable_parameters(self.gpt2_trend)

        self.cnt = 0

        self.num_nodes = configs.num_nodes
        self.rev_in_trend = RevIn(num_features=self.num_nodes)
        self.rev_in_season = RevIn(num_features=self.num_nodes)
        self.rev_in_noise = RevIn(num_features=self.num_nodes)

        self.loss_func = configs.loss_func
        if self.loss_func == "prob":
            # Create output layers for Student's t-distribution parameters
            studentT_output_layer = nn.Linear(configs.pred_len, configs.pred_len)

            self.mu = studentT_output_layer  # Mean
            self.sigma = studentT_output_layer  # Scale (standard deviation)
            self.nu = studentT_output_layer  # Degrees of freedom
        elif self.loss_func == "negative_binomial":
            # Create output layers for Negative Binomial parameters
            negative_binomial_output_layer = nn.Linear(
                configs.pred_len, configs.pred_len
            )

            self.mu = negative_binomial_output_layer  # Mean
            self.alpha = negative_binomial_output_layer

    @classmethod
    def load_pretrained_model(
        cls,
        device,
        cfg=None,
        repo_id="Melady/TEMPO",
        filename="TEMPO-80M_v1.pth",
        cache_dir="./checkpoints/TEMPO_checkpoints",
    ):
        # Download the model checkpoint
        checkpoint_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        )

        # Download the config.json file
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", cache_dir=cache_dir
        )

        # Load the configuration file
        if cfg is None:
            cfg = OmegaConf.load(config_path)

        # Initialize the model
        model = cls(cfg)

        # Construct the full path to the checkpoint
        model_path = os.path.join(cfg.checkpoints, cfg.model_id)
        best_model_path = model_path + "_checkpoint.pth"
        print(f"Loading model from: {best_model_path}")

        # Load the state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        return model

    def store_tensors_in_dict(
        self,
        original_x,
        original_trend,
        original_season,
        original_noise,
        trend_prompts,
        season_prompts,
        noise_prompts,
    ):
        # Assuming prompts are lists of tuples
        self.prompt_record_id += 1
        for i in range(original_x.size(0)):
            self.prompt_record_plot[self.prompt_record_id + i] = {
                "original_x": original_x[i].tolist(),
                "original_trend": original_trend[i].tolist(),
                "original_season": original_season[i].tolist(),
                "original_noise": original_noise[i].tolist(),
                "trend_prompt": trend_prompts[i],
                "season_prompt": season_prompts[i],
                "noise_prompt": noise_prompts[i],
            }

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon)))
        return x * x_inv_norm

    def select_prompt(self, summary, prompt_mask=None):
        prompt_key_matrix = torch.stack(
            tuple([self.prompt_key_dict[i] for i in self.prompt_key_dict.keys()])
        )
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1)  # Pool_size, C
        summary_reshaped = summary.view(-1, self.patch_num)
        summary_mapped = self.summary_map(summary_reshaped)
        summary = summary_mapped.view(-1, 768)
        summary_embed_norm = self.l2_normalize(summary, dim=1)
        similarity = torch.matmul(summary_embed_norm, prompt_norm.t())
        if prompt_mask is not None:
            idx = prompt_mask
        else:
            topk_sim, idx = torch.topk(similarity, k=self.top_k, dim=1)
        if prompt_mask is None:
            count_of_keys = torch.bincount(torch.flatten(idx), minlength=15)
            for i in range(len(count_of_keys)):
                self.prompt_record[f"id_{i}"] += count_of_keys[i].item()

        prompt_value_matrix = torch.stack(
            tuple([self.prompt_value_dict[i] for i in self.prompt_value_dict.keys()])
        )
        batched_prompt_raw = prompt_value_matrix[idx].squeeze(1)
        batch_size, top_k, length, c = batched_prompt_raw.shape  # [16, 3, 5, 768]
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        batched_key_norm = prompt_norm[idx]
        summary_embed_norm = summary_embed_norm.unsqueeze(1)
        sim = batched_key_norm * summary_embed_norm
        reduce_sim = torch.sum(sim) / summary.shape[0]

        # Return the sorted tuple of selected prompts along with batched_prompt and reduce_sim
        selected_prompts = [tuple(sorted(row)) for row in idx.tolist()]
        # print("reduce_sim: ", reduce_sim)

        return batched_prompt, reduce_sim, selected_prompts

    def get_norm(self, x, d="norm"):
        # if d == 'norm':
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x /= stdev

        return x, means, stdev

    def get_patch(self, x):
        x = rearrange(x, "b l m -> b m l")
        x = self.padding_patch_layer(x)  # 4, 1, 420
        x = x.unfold(
            dimension=-1, size=self.patch_size, step=self.stride
        )  # 4,1, 64, 16
        x = rearrange(x, "b m n p -> (b m) n p")  # 4, 64, 16

        return x

    def get_emb(self, x, tokens=None, type="Trend"):
        if tokens is None:
            if type == "Trend":
                x = self.gpt2_trend(inputs_embeds=x).last_hidden_state
            elif type == "Season":
                x = self.gpt2_trend(inputs_embeds=x).last_hidden_state
            elif type == "Residual":
                x = self.gpt2_trend(inputs_embeds=x).last_hidden_state
            return x
        else:
            [a, b, c] = x.shape

            if type == "Trend":
                if self.pool:
                    prompt_x, reduce_sim, selected_prompts_trend = self.select_prompt(
                        x, prompt_mask=None
                    )
                    for selected_prompt_trend in selected_prompts_trend:
                        self.prompt_record_trend[selected_prompt_trend] = (
                            self.prompt_record_trend.get(selected_prompt_trend, 0) + 1
                        )
                    selected_prompts = selected_prompts_trend
                else:
                    prompt_x = self.gpt2_trend.wte(tokens)
                    prompt_x = prompt_x.repeat(a, 1, 1)
                    prompt_x = self.prompt_layer_trend(prompt_x)
                x = torch.cat((prompt_x, x), dim=1)

            elif type == "Season":
                if self.pool:
                    prompt_x, reduce_sim, selected_prompts_season = self.select_prompt(
                        x, prompt_mask=None
                    )
                    for selected_prompt_season in selected_prompts_season:
                        self.prompt_record_season[selected_prompt_season] = (
                            self.prompt_record_season.get(selected_prompt_season, 0) + 1
                        )
                    selected_prompts = selected_prompts_season
                else:
                    prompt_x = self.gpt2_trend.wte(tokens)
                    prompt_x = prompt_x.repeat(a, 1, 1)
                    prompt_x = self.prompt_layer_season(prompt_x)

                x = torch.cat((prompt_x, x), dim=1)
                # x = self.gpt2_trend(inputs_embeds =x_all).last_hidden_state

            elif type == "Residual":
                if self.pool:
                    prompt_x, reduce_sim, selected_prompts_resid = self.select_prompt(
                        x, prompt_mask=None
                    )
                    for selected_prompt_resid in selected_prompts_resid:
                        self.prompt_record_residual[selected_prompt_resid] = (
                            self.prompt_record_residual.get(selected_prompt_resid, 0)
                            + 1
                        )
                    selected_prompts = selected_prompts_resid
                else:
                    prompt_x = self.gpt2_trend.wte(tokens)
                    prompt_x = prompt_x.repeat(a, 1, 1)
                    prompt_x = self.prompt_layer_noise(prompt_x)
                # prompt_x, reduce_sim_trend = self.select_prompt(x, prompt_mask=None)

                x = torch.cat((prompt_x, x), dim=1)

            if self.pool:
                return x, reduce_sim, selected_prompts
            else:
                return x

    def forward(
        self,
        past_target,
        itr=0,
        trend=None,
        season=None,
        noise=None,
        test=False,
    ):
        """
        Computes a forward pass of the TEMPO model.

        Return:
            (mu, sigma, nu), loss_local if model is probabilistic
            outputs, loss_local if model is deterministic
        """
        # Ensure input is three dimensional (batch_size, seq_len, num_features)
        if past_target.dim() < 3:
            past_target = torch.unsqueeze(past_target, -1)

        # Get batch size (B), sequence length (L), and number of features (M)
        B, L, M = past_target.shape

        # Normalize the input time series
        past_target = self.rev_in_trend(past_target, "norm")

        # Compute trend component
        trend_local = self.moving_average(past_target)

        # Perform linear transformation on trend component
        trend_local = self.map_trend(trend_local.squeeze(2)).unsqueeze(2)

        # Compute seaonal component by subtracting the trend from the input
        season_local = past_target - trend_local

        # Perform non-linear transformation on seasonal component
        season_local = self.map_season(season_local.squeeze(2)).unsqueeze(2)

        # Compute residual component
        noise_local = past_target - trend_local - season_local

        # Initialize local loss
        loss_local = None

        if trend is not None:
            trend, _, _ = self.get_norm(trend)
            season, _, _ = self.get_norm(season)
            noise, _, _ = self.get_norm(noise)

            trend_local_l = self.criterion(trend, trend_local)
            season_local_l = self.criterion(season, season_local)
            noise_local_l = self.criterion(noise, noise_local)

            loss_local = trend_local_l + season_local_l + noise_local_l

        # Convert trend, seasonal, and residual components into patches
        trend = self.get_patch(trend_local)
        season = self.get_patch(season_local)
        noise = self.get_patch(noise_local)

        # Project patches into hidden space
        trend = self.in_layer_trend(trend)
        season = self.in_layer_season(season)
        noise = self.in_layer_noise(noise)

        if self.is_gpt and self.prompt:
            trend = self.get_emb(trend, self.gpt2_trend_token["input_ids"], "Trend")
            season = self.get_emb(season, self.gpt2_season_token["input_ids"], "Season")
            noise = self.get_emb(
                noise, self.gpt2_residual_token["input_ids"], "Residual"
            )
        else:
            trend = self.get_emb(trend)
            season = self.get_emb(season)
            noise = self.get_emb(noise)

        # Concatenate trend, seasonal, and residual component embeddings
        past_target_all = torch.cat((trend, season, noise), dim=1)

        # Perform forward pass through GPT-2 model
        past_target = self.gpt2_trend(inputs_embeds=past_target_all).last_hidden_state

        if self.prompt:
            trend = past_target[:, : self.token_len + self.patch_num, :]
            season = past_target[
                :,
                self.token_len
                + self.patch_num : 2 * self.token_len
                + 2 * self.patch_num,
                :,
            ]
            noise = past_target[:, 2 * self.token_len + 2 * self.patch_num :, :]
            if not self.use_token:
                trend = trend[:, self.token_len :, :]
                season = season[:, self.token_len :, :]
                noise = noise[:, self.token_len :, :]
        else:
            trend = past_target[:, : self.patch_num, :]
            season = past_target[:, self.patch_num : 2 * self.patch_num, :]
            noise = past_target[:, 2 * self.patch_num :, :]

        # Get predicted trend, seasonal, and residual components
        # Here, the components have shape (B * M, pred_len)
        trend = self.out_layer_trend(trend.reshape(B * M, -1))
        season = self.out_layer_season(season.reshape(B * M, -1))
        noise = self.out_layer_noise(noise.reshape(B * M, -1))

        # Reshape components from (B * M, pred_len) to (B, pred_len, M)
        trend = rearrange(trend, "(b m) l -> b l m", b=B)
        season = rearrange(season, "(b m) l -> b l m", b=B)
        noise = rearrange(noise, "(b m) l -> b l m", b=B)

        # Construct predicted forecast by adding trend, season, and residual
        outputs = trend + season + noise

        # Denormalize output
        outputs = self.rev_in_trend(outputs, "denorm")

        # If loss function is Student's t-distribution NLL:
        if self.loss_func == "prob":
            # Set distribution output to Student's t-distribution
            distr_output = StudentTOutput()

            # Create layer to project outputs to Student's t parameters
            args_proj = distr_output.get_args_proj(M)

            # Convert outputs into Student's t-distribution parameters
            distr_args = args_proj(outputs)  # (mu, sigma, nu)

            mu = distr_args[0]

            prediction_length = mu.size()[0]

            loc = torch.zeros(prediction_length, 1)
            scale = torch.zeros(prediction_length, 1)

            return distr_args, loc, scale

        # if self.loss_func == "prob":
        #     outputs = rearrange(outputs, "b l m-> b m l", b=B).squeeze()

        #     print(f"Loss func: {self.loss_func}")
        #     print(f'outputs shape: {outputs.shape}')

        #     mu = self.mu(outputs)
        #     sigma = F.softplus(self.sigma(outputs)) + 1e-6  # Ensure scale is positive
        #     nu = F.softplus(self.nu(outputs)) + 2  # Ensure degrees of freedom > 2
        #     student_T_arguments = (
        #         mu,  # Location
        #         sigma,  # Scale
        #         nu,  # Degrees of freedom
        #     )  # Parameters for student's t-distribution
        #     print(f'mu shape: {mu.shape}')
        #     print(f'sigma shape: {sigma.shape}')
        #     print(f'nu shape: {nu.shape}')
        #     if test:
        #         return student_T_arguments, loss_local
        #     return student_T_arguments, loss_local

        # elif self.loss_func == "negative_binomial":
        #     mu = F.softplus(self.mu(past_target)) + 1e-4  # Ensure mean is positive
        #     alpha = (
        #         F.softplus(self.alpha(past_target)) + 1e-4
        #     )  # Ensure dispersion is positive
        #     if test:
        #         return (
        #             mu.permute(0, 2, 1),
        #             alpha.permute(0, 2, 1),
        #         ), None  # Return to [Batch, Output length, Channel]
        #     else:
        #         return (mu.permute(0, 2, 1), alpha.permute(0, 2, 1)), loss_local

        # if test:
        #     return outputs, loss_local
        # return outputs, loss_local

    def set_to_target_length(self, x):
        """
        Pads/truncates x so it has self.seq_len time steps

        Args:
        - x: Input time series data (shape: [B, L, M])

        Returns:
        - Padded/truncated version of x
        """
        # Maximum supported number of time steps in x
        target_length = self.seq_len

        # Batch size, length, and number of channels
        x = torch.squeeze(x, dim=0)
        x = torch.squeeze(x, dim=1)
        B, L, M = x.shape

        # If x is longer than target_length, then truncate it to the last
        # target_length time steps
        if L > target_length:
            warnings.warn(
                f"Input length {L} is larger than the maximum supported length of {target_length}. "
                f"This may influence performance. Cutting the input to the last {target_length} time steps."
            )

            return x[:, -target_length:, :]

        # remaining number of time steps for x to have target_length time steps
        pad_length = target_length - L

        warnings.warn(
            f"Input length {L} is smaller than the required length of {target_length}. "
            f"The time series has been {'repeated' if pad_length <= L else 'zero-padded'} to reach the required length."
        )

        if pad_length <= L:
            # Pad by repeating the time series
            x_padded = torch.cat([x] * (target_length // L + 1), dim=1)[
                :, :target_length, :
            ]
        else:
            # Pad with zeros at the beginning
            padding = torch.zeros(B, pad_length, M)
            x_padded = torch.cat([padding, x], dim=1)

        return x_padded

    def predict(self, x, pred_length=96):
        """
        Computes predictions for pred_length future time steps using the TEMPO
        model and given input data.

        Args:
        - x: Input time series data (shape: [B, L, M])
            - B: batch size
            - L: length
            - M: number of channels

        Returns:
        - Predicted values for pred_length future time steps
        """
        # Set model to evaluation mode
        self.eval()

        # Set x's shape to [1, 336, 1]
        x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(2)

        # Normalize x
        x = self.rev_in_trend(x, "norm")

        # Pad/truncate x so it has self.seq_len time steps
        x = self.set_to_target_length(x)

        # Compute predictions for future time steps
        with torch.no_grad():
            current_input = x.clone()
            all_predictions = []  # List of predictions

            while len(all_predictions) < pred_length:
                # Forward pass
                outputs, _ = self.forward(current_input, test=True)
                outputs = self.rev_in_trend(outputs, "denorm")
                step_size = outputs.shape[1]

                # Extract the predicted values
                predicted_values = outputs.cpu().squeeze().numpy()[-step_size:]

                # Append to all predictions
                all_predictions.extend(predicted_values)

                # Update the input for the next iteration
                new_sequence = np.concatenate(
                    [
                        current_input.cpu().squeeze().numpy()[step_size:],
                        predicted_values,
                    ]
                )
                current_input = (
                    torch.FloatTensor(new_sequence).unsqueeze(0).unsqueeze(2)
                )

        # Trim to the desired length
        return np.array(all_predictions[:pred_length])

    def predict_prob(
        self,
        x,
        num_samples,
        trend=None,
        seasonal=None,
        residual=None,
        pred_len=96,
    ):
        """
        Computes probabilistic predictions for pred_len future time steps
        using the TEMPO model.

        Args:
        - x: Input time series data (shape: [B, L, M])
            - B: batch size
            - L: length
            - M: number of channels
        - num_samples: Number of samples to use when computing distributions
        - trend: Trend component
        - seasonal: Seasonal component
        - residual: Residual component
        - pred_len: Number of future time steps to compute distributions for

        Returns:
        - Probability distribution for each of the pred_len future time steps
        """
        # Forward pass
        outputs, _ = self.forward(x, trend=trend, season=seasonal, noise=residual)

        # Compute probabilistic forecasts
        if self.loss_func == "prob":
            # Get Student's t-distribution parameters
            mu, sigma, nu = outputs

            # Create Student's t-distribution
            student_t = dist.StudentT(df=nu, loc=mu, scale=sigma)

            # Generate num_samples samples for each prediction
            probabilistic_forecast = student_t.rsample((num_samples,))

        elif self.loss_func == "negative_binomial":
            # Get Negative Binomial distribution parameters
            mu, alpha = outputs

            # Generate num_samples samples for each prediction
            probabilistic_forecast = sample_negative_binomial(mu, alpha, num_samples)

        return probabilistic_forecast

    def get_predictor(self, input_transform, prediction_length=96, batch_size=128):
        """
        Returns a GluonTS PyTorch predictor for performing inference.
        """
        return PyTorchPredictor(
            prediction_length=prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
        )

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params} || All params: {all_param}")
        print(f"Trainable params: {trainable_params} || All params: {all_param}")
        print(f"Trainable params: {trainable_params} || All params: {all_param}")
