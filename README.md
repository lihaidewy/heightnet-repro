# HeightNet / BEVHeight reproduction

PyTorch implementation adapted to the S3S and S3M roadside datasets.

- `src/heightnet/`: model, geometry, BEV pooling, detection and radar modules.
- `scripts/`: training, evaluation, visualization and module checks.
- `docs/`: geometry conventions and recorded results.
- `outputs/`: experiment configurations, logs and visualizations.

Raw datasets and model checkpoints are excluded. To train, restore S3S/S3M images, labels and radar files under `data/`, and provide dataset split JSON files.

Training entry point: `python -m scripts.train_heightnet_full`. Dependencies include PyTorch, torchvision, NumPy, Pillow, pandas and SciPy; versions are not pinned.
