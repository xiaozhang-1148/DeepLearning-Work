import subprocess
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from pytorch_lightning.utilities.cli import LightningCLI
from posformer.datamodule import CROHMEDatamodule
from posformer.core.processors import LitPosFormer

from pytorch_lightning import Trainer

class MyLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_process = None

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for the model')
    
    def before_fit(self):
        # --- 简化: 移除所有手动配置，这些都应由 config.yaml 控制 ---
        # 清理日志目录的逻辑可以保留
        if self.trainer.is_global_zero:
            log_dir = self.trainer.logger.log_dir
            # 仅当不是从检查点恢复时才清理
            if self.config['ckpt_path'] is None and Path(log_dir).exists():
                print(f"--- Cleaning up old log directory: {log_dir} ---")
                shutil.rmtree(log_dir)
                Path(log_dir).mkdir(parents=True, exist_ok=True)

        # --- 自动启动 TensorBoard 的逻辑可以保留 ---
        if self.trainer.is_global_zero:
            log_dir = self.trainer.logger.log_dir
            print(f"--- Starting TensorBoard on logdir: {log_dir} ---")
            
            tb_log_file = open("tensorboard_launch.log", "w")

            self.tb_process = subprocess.Popen(
                ['tensorboard', '--logdir', log_dir, '--port', '6008', '--reload_interval', '15'],
                stdout=tb_log_file,
                stderr=subprocess.STDOUT
            )
            print("--- TensorBoard process started. See tensorboard_launch.log for details. ---")
            print(f"--- Access at http://<YOUR_SERVER_IP>:6008/ or use SSH forwarding. ---")

    def after_fit(self):
        if self.trainer.is_global_zero and hasattr(self, 'tb_process') and self.tb_process:
            print("--- Shutting down TensorBoard ---")
            self.tb_process.terminate()
            self.tb_process.wait()

    def after_test(self):
        if self.trainer.is_global_zero and hasattr(self, 'tb_process') and self.tb_process:
            print("--- Shutting down TensorBoard ---")
            self.tb_process.terminate()
            self.tb_process.wait()


cli = MyLightningCLI(
    LitPosFormer,
    CROHMEDatamodule,
    save_config_overwrite=True,
)