from setuptools import setup

setup(
    name="mmcr",
    packages=["mmcr"],
    install_requires=[
        'librosa >= 0.8',
        'torchaudio_augmentations==0.2.1', # for augmentation
        'numpy',
        'pandas',
        'einops',
        'sklearn',
        'wandb',
        'jupyter',
        'matplotlib',
        'omegaconf',
        'astropy',
        'transformers',
    ]
)