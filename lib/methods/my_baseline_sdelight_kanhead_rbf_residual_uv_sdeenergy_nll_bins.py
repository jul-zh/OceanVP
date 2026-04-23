import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from lib.models import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_UV
from lib.utils import reduce_tensor
from .base_method import Base_method


class MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_UV_SDEENERGY_NLL_BINS(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)

        self.mse_criterion = nn.MSELoss()
        self.criterion = self.mse_criterion

        self.lambda_sde = getattr(args, "lambda_sde", 5e-5)
        self.sde_eps = getattr(args, "sde_eps", 1e-6)

        bins_u_path = getattr(
            args,
            "sde_bins_u_path",
            "/home/yzhidkova/projects/OceanVP/logs/sde_bins_u0_train_1000.json"
        )
        bins_v_path = getattr(
            args,
            "sde_bins_v_path",
            "/home/yzhidkova/projects/OceanVP/logs/sde_bins_v0_train_1000.json"
        )

        with open(bins_u_path, "r", encoding="utf-8") as f:
            data_u = json.load(f)
        with open(bins_v_path, "r", encoding="utf-8") as f:
            data_v = json.load(f)

        self.bin_edges_u = torch.tensor(data_u["edges"], dtype=torch.float32, device=device)
        self.bin_drift_u = torch.tensor(
            [0.0 if v is None else float(v) for v in data_u["drift"]],
            dtype=torch.float32,
            device=device
        )
        self.bin_diff2_u = torch.tensor(
            [1.0 if v is None else max(float(v), self.sde_eps) for v in data_u["diffusion2"]],
            dtype=torch.float32,
            device=device
        )

        self.bin_edges_v = torch.tensor(data_v["edges"], dtype=torch.float32, device=device)
        self.bin_drift_v = torch.tensor(
            [0.0 if v is None else float(v) for v in data_v["drift"]],
            dtype=torch.float32,
            device=device
        )
        self.bin_diff2_v = torch.tensor(
            [1.0 if v is None else max(float(v), self.sde_eps) for v in data_v["diffusion2"]],
            dtype=torch.float32,
            device=device
        )

    def _build_model(self, args):
        return MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_UV(**args).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        else:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)
            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def _lookup_single(self, x, edges, drift_table, diff2_table):
        flat = x.reshape(-1)
        idx = torch.bucketize(flat, edges[1:-1])
        drift = drift_table[idx].reshape_as(x)
        diff2 = diff2_table[idx].reshape_as(x)
        return drift, diff2

    def _lookup_uv_bins(self, x):
        # x: [B,2,H,W]
        x_u = x[:, 0:1]
        x_v = x[:, 1:2]

        drift_u, diff2_u = self._lookup_single(x_u, self.bin_edges_u, self.bin_drift_u, self.bin_diff2_u)
        drift_v, diff2_v = self._lookup_single(x_v, self.bin_edges_v, self.bin_drift_v, self.bin_diff2_v)

        drift = torch.cat([drift_u, drift_v], dim=1)
        diff2 = torch.cat([diff2_u, diff2_v], dim=1)
        return drift, diff2

    def _sde_energy_loss(self, batch_x, pred_y):
        x_last = batch_x[:, -1]   # [B,2,H,W]
        total = 0.0
        nsteps = 0

        prev_state = x_last
        for t in range(pred_y.shape[1]):
            cur_state = pred_y[:, t]          # [B,2,H,W]
            delta = cur_state - prev_state    # [B,2,H,W]

            drift, diff2 = self._lookup_uv_bins(prev_state)
            resid = delta - drift

            step_loss = resid.pow(2) / (diff2 + self.sde_eps) + torch.log(diff2 + self.sde_eps)
            total = total + step_loss.mean()

            prev_state = cur_state
            nsteps += 1

        return total / max(nsteps, 1)

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)

        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader
        end = time.time()

        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            runner.call_hook('before_train_iter')
            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                mse_loss = self.mse_criterion(pred_y, batch_y)
                sde_loss = self._sde_energy_loss(batch_x, pred_y)
                loss = mse_loss + self.lambda_sde * sde_loss

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad,
                    clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters()
                )
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1
            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))
            if not self.by_epoch:
                self.scheduler.step()

            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                train_pbar.set_description(
                    f"train loss: {loss.item():.4f} | mse: {mse_loss.item():.4f} | sde_nll: {sde_loss.item():.4f} | data time: {data_time_m.avg:.4f}"
                )
            end = time.time()

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, losses_m, eta
