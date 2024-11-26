# Installation


Follow these intructions

1. Clone our repository
```
git clone https://github.com/sijang/SpachTransformer.git
cd SpachTransformer
```

2-1. Make conda environment (option 1)
```
conda env create -f env.yml
conda activate spach
```

2-2. Make conda environment (option 2)
```
conda create -n spach python=3.9
conda activate spach
pip install spach-transformer
pip install --upgrade torch torchvision torchaudio
```

