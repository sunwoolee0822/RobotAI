# robot_ai2

설문 + 웨어러블 센서 시계열을 이용한 이진 분류. CoFormer 백본 위에 PyTorch Lightning
학습 파이프라인과 분석 코드를 붙였다.

## 외부 의존성: CoFormer

**이 저장소에는 CoFormer 코드가 포함되어 있지 않다.** 별도로 clone 해야 한다.

```bash
git clone https://github.com/MediaBrain-SJTU/CoFormer.git
cd CoFormer && git checkout 69261db && cd ..
```

프로젝트 루트에 `CoFormer/` 디렉토리로 두면 된다 (`train.py`가 `sys.path`에 자동으로 추가함).

```
robot_ai2/
├── CoFormer/        ← clone 해서 여기에
├── train.py
├── module.py
└── data/
```

**원본 코드는 수정 없이 그대로 사용한다.** 실제로 쓰는 것은 백본 아키텍처 하나뿐:

```python
from CoFormer.models.model_medical_attn_aggre import make_model
```

`models/model_medical_attn_aggre.py`는 CoFormer 내부의 다른 모듈을 import 하지 않는
self-contained 파일이라, 나머지(`train_medical.py`, `data/dataloader.py`, `utils/*`)는
사용하지 않는다. 학습 루프·데이터로더·평가·분석은 전부 이 저장소의 코드다.

> CoFormer는 LICENSE 파일이 없어 재배포하지 않고 clone 방식으로 참조한다.
> 원저작권은 MediaBrain-SJTU에 있다.

## 환경

검증된 조합 (Python 3.10):

| 패키지 | 버전 |
|---|---|
| torch | 2.4.0+cu121 |
| dgl | 2.4.0+cu121 |
| pytorch-lightning | 2.6.4 |
| torchmetrics | 1.9.0 |

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install pytorch-lightning torchmetrics wandb pyyaml numpy pandas scikit-learn matplotlib tqdm
```

`dgl`은 필수다. 백본이 GAT 레이어(`GATv2Conv`, `KNNGraph`)를 항상 생성하므로 끌 수 없다.

## 데이터

원시 데이터는 개인정보 문제로 저장소에 포함하지 않는다. 전처리 파이프라인만 포함:

```bash
python data/step1_prepare.py    # 설문 4종 + 센서 CSV → 피험자별 DataFrame 캐시 (1~2시간)
python data/step2_coformer.py   # window 72slot(3일), stride 24slot(1일) → npy 텐서
```

`step2` 출력 (`data/numpy_all_chunk_72_24feat/`):

| 파일 | 내용 |
|---|---|
| `array.npy` | 센서 시계열 |
| `time.npy` | 타임스탬프 (불규칙 간격 인코딩용) |
| `static.npy` | 정적 피처 |
| `mask.npy` | 결측 마스크 |
| `gt.npy` | 라벨 (0/1) |
| `split.npy` | `[train_idx, val_idx, test_idx]` |

## 학습

```bash
python train.py --cfg cfg/robotai.yaml --gpu 0 --wandb_name my_run
```

주요 옵션:

```
--data_root           npy 디렉토리
--split_path          split.npy 경로
--num_layers 8        인코더 레이어 수
--d_model 256         모델 차원
--num_neighbors 30    GAT KNN 이웃 수
--val_check_interval 625   step 단위 validation
--resume <ckpt>       체크포인트에서 이어서 학습
--no_wandb            wandb 끄기
```

`val_auroc` 기준으로 best 3개 체크포인트를 저장하고, patience 25로 early stopping.
학습 후 best 체크포인트로 test를 돌려 `test_predictions_{wandb_name}.npz`에
`pred / prob / gt`를 저장한다.

하이퍼파라미터는 [cfg/robotai.yaml](cfg/robotai.yaml)에서 관리한다 (seed, batch_size, lr,
weight_decay, lr_factor, patience).

클래스 불균형은 `WeightedRandomSampler`로 1:1 oversampling 한다
([data/dataset.py](data/dataset.py)).

## 구성

| 파일 | 역할 |
|---|---|
| `train.py` | 학습 엔트리포인트 (Lightning Trainer) |
| `module.py` | LightningModule — AdamW + ReduceLROnPlateau, 6종 metric (f1/auroc/auprc/acc/prec/rec) |
| `data/dataset.py` | Dataset / DataModule, oversampling sampler |
| `data/step1_prepare.py`, `data/step2_coformer.py` | 전처리 |
| `make_group_splits.py` | 결측률 high/low 그룹별 split 생성 |
| `make_missing_*.py` | 결측 시나리오별 test split 생성 |
| `analysis*.py`, `plot1.py`, `make_figures.py` | 결과 분석 및 그림 |
| `extra_attention.py` | 어텐션 맵 시각화 |

## 참고

- CoFormer: https://github.com/MediaBrain-SJTU/CoFormer



아으-------