import zipfile
from typing import List , Tuple
import time
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.nn import functional as F
from torchvision.utils import make_grid
from Pos_Former.datamodule import Batch, vocab ,label_make_muti
from Pos_Former.model.posformer import PosFormer
from Pos_Former.utils.utils import (ExpRateRecorder, Hypothesis,ce_loss_all,ce_loss,
                               to_bi_tgt_out)
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitPosFormer(pl.LightningModule):
    """
    A PyTorch Lightning module for PosFormer.
    """

    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = PosFormer(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.exprate_recorder = ExpRateRecorder()
        self.summary_writer = None # 由 train.py 注入

    """ 以下部分，均为用于构建训练监视器内容的钩子函数 """
    """ 辅助功能：用于在训练开始时，记录数据集样本和模型图。"""
    def on_fit_start(self):
        
        if self.global_rank == 0 and self.logger and self.trainer.datamodule:
            # 1. 记录一个批次的训练图像
            # 通过 datamodule 获取 train_dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            try:
                batch = next(iter(train_loader))
            except StopIteration:
                print("Warning: Train dataloader is empty, cannot log image samples.")
                return

            # 将图像张量制作成网格
            grid = make_grid(batch.imgs, nrow=4, normalize=True)
            self.logger.experiment.add_image('train_dataset_samples', grid, self.global_step)
            
            # 2. 记录模型图 (确保输入维度正确)
            # 使用一个 dummy_input 来追踪模型图
            dummy_tgt = torch.zeros((batch.imgs.size(0), self.hparams.max_len), dtype=torch.long, device=self.device)
            try:
                # 使用 self.model，并传入 batch.imgs 和 batch.mask
                self.logger.experiment.add_graph(self.model, (batch.imgs, batch.mask, dummy_tgt))
            except Exception as e:
                print(f"Could not add model graph to TensorBoard: {e}")
    def on_epoch_end(self):
        """
        在每个 epoch 结束时，记录模型参数的直方图。
        此功能会产生大量日志，可以注释掉以保持 TensorBoard 清洁。
        """
        if self.global_rank == 0 and self.logger:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
        pass
    def on_validation_epoch_start(self):
        """
        在每个验证 epoch 开始时，记录模型参数的直方图。
        """
        if self.global_rank == 0 and self.logger:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch, 50)
        pass

    """ 调用model → 完成模型训练部分 """
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.model(img, img_mask, tgt, logger)

    def training_step(self, batch: Batch, _):      
        # avoid variable name collision: keep model outputs and targets separate
        tgt, tgt_out = to_bi_tgt_out(batch.indices, self.device)   # tgt: input for decoder, tgt_out: ground-truth tensor
        out_hat, out_hat_layer, out_hat_pos = self(batch.imgs, batch.mask, tgt, self.trainer.logger)
        tgt_list = tgt.cpu().numpy().tolist()
        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor = torch.LongTensor(layer_num).to(self.device)   #[2b,l,5]
        final_pos_tensor = torch.LongTensor(final_pos).to(self.device)   #[2b,l,6]
        # use tgt_out (ground-truth) as target for loss computation
        loss, layer_loss, pos_loss = ce_loss_all(out_hat, tgt_out, out_hat_layer, layer_num_tensor, out_hat_pos, final_pos_tensor)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, sync_dist=True,prog_bar=True)
        self.log("train_loss_pos",pos_loss, logger=True, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log("train_loss_layernum",layer_loss, logger=True,on_step=False, on_epoch=True, sync_dist=True,prog_bar=True)
        loss = (2 * loss + 0.1 * layer_loss + 0.1 * pos_loss) / 2.2
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        tgt, tgt_out = to_bi_tgt_out(batch.indices, self.device)
        out_hat, out_hat_layer, out_hat_pos = self(batch.imgs, batch.mask, tgt, self.trainer.logger)

        tgt_list = tgt.cpu().numpy().tolist()
        layer_num, final_pos = label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor = torch.LongTensor(layer_num).to(self.device)   #[2b,l,5]
        final_pos_tensor = torch.LongTensor(final_pos).to(self.device)   #[2b,l,6]

        loss, layer_loss, pos_loss = ce_loss_all(out_hat, tgt_out, out_hat_layer, layer_num_tensor, out_hat_pos, final_pos_tensor)

        self.log(
            "val_loss",
            loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_pos",
            pos_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_layernum",
            layer_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        if batch_idx == 0 and self.global_rank == 0 and self.logger:
            # 从批次中获取第一个样本的真实标签和预测结果
            ground_truth_indices = batch.indices[0]
            prediction_indices = hyps[0].seq

            # 将索引转换为可读的文本
            ground_truth_text = vocab.indices2label(ground_truth_indices)
            prediction_text = vocab.indices2label(prediction_indices)

            # 格式化为 Markdown 文本
            log_text = (
                f"**Epoch: {self.current_epoch}**\n\n"
                f"**Ground Truth:**\n\n---\n\n`{ground_truth_text}`\n\n"
                f"**Prediction:**\n\n---\n\n`{prediction_text}`"
            )

            # 将文本添加到 TensorBoard 的 "TEXT" 选项卡
            self.logger.experiment.add_text(
                "validation_sample_comparison", log_text, self.current_epoch
            )

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            logger=True,
           
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        start_time = time.time()  # Start timing
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        inference_time = time.time() - start_time  # Compute inference time for this batch
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log('batch_inference_time', inference_time)  # Optional: log inference time per batch
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps], inference_time

    def test_epoch_end(self, test_outputs) -> None:
        total_inference_time = sum (output[2] for output in test_outputs)  # Sum up the inference times
        print(f"Total Inference Time: {total_inference_time} seconds")

        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, _ in test_outputs:  # Unpack the ignored time measurements
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.model.beam_search(img, mask, **self.hparams)


    # 原模型使用 SGD 优化器 + learning_Rate 0.08 → 
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。
        可通过注释切换来对比 AdamW 和 SGD。
        """
        # --- 1. 选择你的优化器 ---
        # 方案 A: AdamW (更常用，通常效果更好)  →  AdamW 经过 100 epoch，准确率为 59.87 %
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.learning_rate,
            weight_decay=1e-4
        )
        
        # 方案 B: SGD (需要更精细的调度策略) →  SGD 经过 300 epoch，准确率为 54.33 %
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     momentum=0.9,
        #     weight_decay=1e-4
        # )

        # --- 2. 配置学习率调度器 (使用线性热身 + 余弦退火) ---
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            total_steps = self.trainer.max_epochs * len(train_loader)
        except Exception:
            # 如果无法获取 dataloader，则回退到简单调度器
            return optimizer

        # 设置热身步数，例如总步数的 5%
        warmup_steps = int(total_steps * 0.05)

        def lr_lambda(current_step):
            # 线性热身阶段
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # 余弦退火阶段
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # 使用 math.cos 替代 torch.cos 以避免设备问题
            import math
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 必须在每一步更新
                "frequency": 1,
            },
        }
