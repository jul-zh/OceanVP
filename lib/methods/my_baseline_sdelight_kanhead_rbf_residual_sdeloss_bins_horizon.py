import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from lib.models import MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL
from lib.utils import reduce_tensor
from .base_method import Base_method


class MY_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL_SDELOSS_BINS_HORIZON(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

        self.lambda_sde = getattr(args, "lambda_sde", 1e-4)

        weights = getattr(args, "horizon_weights", None)
        if weights is None:
            weights = [1.0]*4 + [1.2]*4 + [1.4]*4 + [1.6]*4
        self.horizon_weights = torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)

        bins_path = getattr(args, "sde_bins_path", "/home/yzhidkova/projects/OceanVP/logs/sde_bins_t0_train_1000.json")
        with open(bins_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.bin_edges = torch.tensor(data["edges"], dtype=torch.float32, device=device)
        self.bin_drift = torch.tensor([0.0 if v is None else v for v in data["drift"]], dtype=torch.float32, device=device)

    def _build_model(self, args):
        return MY_OCEAN_BASELINE_SDELIGHT_KANHEAD_RBF_RESIDUAL(**args).to(self.device)

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

    def _lookup_bin_drift(self, x):
        flat = x.reshape(-1)
        idx = torch.bucketize(flat, self.bin_edges[1:-1])
        return self.bin_drift[idx].reshape_as(x)

    def _sde_loss(self, batch_x, pred_y):
        x_last = batch_x[:, -1]
        total = 0.0
        count = 0
        prev_state = x_last
        for t in range(pred_y.shape[1]):
            cur_state = pred_y[:, t]
            delta_pred = cur_state - prev_state
            target_drift = self._lookup_bin_drift(prev_state)
            total = total + ((delta_pred - target_drift) ** 2).mean()
            prev_state = cur_state
            count += 1
        return total / max(count, 1)

    def _weighted_mse(self, pred_y, batch_y):
        sq = (pred_y - batch_y) ** 2
        return (sq * self.horizon_weights).mean()

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
                mse_loss = self._weighted_mse(pred_y, batch_y)
                sde_loss = self._sde_loss(batch_x, pred_y)
                loss = mse_loss + self.lambda_sde * sde_loss

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(loss, self.model_optim,
                                 clip_grad=self.args.clip_grad,
                                 clip_mode=self.args.clip_mode,
                                 parameters=self.model.parameters())
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
                    f"train loss: {loss.item():.4f} | wmse: {mse_loss.item():.4f} | sde: {sde_loss.item():.4f} | data time: {data_time_m.avg:.4f}"
                )
            end = time.time()

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, losses_m, eta
