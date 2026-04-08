import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from lib.models import MY_OCEAN_BASELINE_RBFKAN_PROB
from lib.utils import reduce_tensor
from .base_method import Base_method


class MY_BASELINE_RBFKAN_PROB(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)

    def _build_model(self, args):
        return MY_OCEAN_BASELINE_RBFKAN_PROB(**args).to(self.device)

    @staticmethod
    def gaussian_nll(mu, raw_sigma, target):
        sigma = F.softplus(raw_sigma) + 1e-4
        loss = 0.5 * (2.0 * torch.log(sigma) + ((target - mu) ** 2) / (sigma ** 2))
        return loss.mean()

    def _predict(self, batch_x, batch_y=None, **kwargs):
        mu, raw_sigma = self.model(batch_x)
        return mu

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
                mu, raw_sigma = self.model(batch_x)
                loss = self.gaussian_nll(mu, raw_sigma, batch_y)

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
                log_buffer = f'train prob loss: {loss.item():.4f} | data time: {data_time_m.avg:.4f}'
                train_pbar.set_description(log_buffer)

            end = time.time()

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
