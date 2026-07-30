import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, 'CoFormer'))

from CoFormer.models.model_medical_attn_aggre import make_model
from data.dataset import RobotAIDataModule
from module import CoFormerModule

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--cfg', default='cfg/robotai.yaml')
    parser.add_argument('--gpu', type=int, default=0)

    # model
    parser.add_argument('--input_dim',          type=int,   default=1)
    parser.add_argument('--output_dim',         type=int,   default=2)
    parser.add_argument('--num_layers',         type=int,   default=8)
    parser.add_argument('--heads',              type=int,   default=8)
    parser.add_argument('--d_model',            type=int,   default=256)
    parser.add_argument('--d_ff',               type=int,   default=256)
    parser.add_argument('--dropout',            type=float, default=0.1)
    parser.add_argument('--num_agents',         type=int,   default=None)
    parser.add_argument('--num_neighbors',      type=int,   default=30)
    parser.add_argument('--agent_encoding_dim', type=int,   default=32)
    parser.add_argument('--static_dim',         type=int,   default=4)

    # data
    parser.add_argument('--data_root', default='/home/hail/robot_ai2/data/numpy_all_chunk_72/')
    parser.add_argument('--split_path', default='/home/hail/robot_ai2/data/numpy_all_chunk_72/split.npy')

    # training (step-wise)
    parser.add_argument('--val_check_interval', type=int, default=625)   # step-wise; batch16 기준 ~8 val/epoch (baseline batch4의 2500과 같은 cadence)
    parser.add_argument('--log_every_n_steps', type=int, default=50)

    # wandb
    parser.add_argument('--wandb_name',    default= 'coformer_new')
    parser.add_argument('--wandb_project', default='coformer_new')
    parser.add_argument('--no_wandb',      action='store_true')
    parser.add_argument('--pred_out',      default=None,
                        help='test 예측 저장 경로 (기본 test_predictions_{wandb_name}.npz)')
    parser.add_argument('--resume',        default=None,
                        help='체크포인트 경로에서 이어서 학습 (trainer.fit ckpt_path)')

    args = parser.parse_args()

    # cfg
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # 실험별 예측 결과 파일 분리 (덮어쓰기 방지)
    cfg['pred_out'] = args.pred_out or f'test_predictions_{args.wandb_name}.npz'

    # seed
    pl.seed_everything(cfg['seed'])

    # data
    dm = RobotAIDataModule(
        data_root=args.data_root,
        split_path=args.split_path,
        batch_size=cfg['batch_size'],
    )
    dm.setup()

    if args.num_agents is None:
        args.num_agents = dm.train_dset.data.shape[1]

    # model
    backbone = make_model(
        src_vocab=args.input_dim,
        tgt_vocab=args.output_dim,
        N=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.heads,
        dropout=args.dropout,
        num_agents=args.num_agents,
        num_neighbors=args.num_neighbors,
        agent_encoding_dim=args.agent_encoding_dim,
        static_dim=args.static_dim,
    )
    model = CoFormerModule(backbone, cfg)

    # callbacks
    ckpt_callback = ModelCheckpoint(
        monitor='val_auroc',       # step-wise니까 _epoch suffix 없음
        save_top_k=3,             # best 3개 저장
        mode='max',
        filename='{epoch}-{step}-{val_loss:.4f}',
    )
    es_callback = EarlyStopping(
        monitor='val_auroc',
        mode='max',
        patience=25,
    )

    # wandb
    loggers = []
    if not args.no_wandb:
        run_name = args.wandb_name or 'coformer_robotai'
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            group='coformer_robotai',
            save_dir='logs',
            config={
                **cfg,
                'num_layers':         args.num_layers,
                'heads':              args.heads,
                'd_model':            args.d_model,
                'd_ff':               args.d_ff,
                'dropout':            args.dropout,
                'num_neighbors':      args.num_neighbors,
                'agent_encoding_dim': args.agent_encoding_dim,
                'static_dim':         args.static_dim,
                'num_agents':         args.num_agents,
                'val_check_interval': args.val_check_interval,
            }
        )
        loggers.append(wandb_logger)

    # trainer (step-wise 로깅)
    trainer = pl.Trainer(
        devices=[args.gpu],
        accelerator='gpu',
        precision='32',
        max_epochs=cfg['num_epochs'],
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
        val_check_interval=args.val_check_interval,   # ← step-wise
        log_every_n_steps=args.log_every_n_steps,     # ← step-wise
        callbacks=[ckpt_callback, es_callback],
        logger=loggers if loggers else True,
        enable_checkpointing=True,
    )

    # train (--resume 시 체크포인트에서 epoch/optimizer/LR 상태 복원)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader(),
                ckpt_path=args.resume)

    # test (best ckpt)
    best_model = CoFormerModule.load_from_checkpoint(
        ckpt_callback.best_model_path,
        model=make_model(
            src_vocab=args.input_dim,
            tgt_vocab=args.output_dim,
            N=args.num_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            h=args.heads,
            dropout=args.dropout,
            num_agents=args.num_agents,
            num_neighbors=args.num_neighbors,
            agent_encoding_dim=args.agent_encoding_dim,
            static_dim=args.static_dim,
        ),
        cfg=cfg,
    )
    trainer.test(best_model, dm.test_dataloader())

    if not args.no_wandb:
        wandb_logger.experiment.finish()
