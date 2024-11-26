from setuptools import setup, find_packages

setup(
    name='spach-transformer',   
    version='0.1.0',
    author='sijang',
    author_email='sein.jang@yale.edu',
    description='A package for SpachTransformer and related models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sijang/SpachTransformer',  
    packages=find_packages(),  # Automatically includes all packages (like `model`)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.1",
        "torchvision>=0.11.2",
        "torchaudio>=0.10.1",
        "numpy>=1.21.2",
        "scipy>=1.7.3",
        "h5py>=3.6.0",
        "imageio>=2.13.5",
        "matplotlib>=3.5.1",
        "scikit-image>=0.19.1",
        "scikit-learn>=1.0.2",
        "simpleitk>=2.2.1",
        "tqdm>=4.62.3",
        "opencv-python>=4.5.5.62",
        "pyyaml>=6.0",
        "timm>=0.4.12",
        "einops>=0.3.2",
        "torchmetrics>=0.11.0",
        "yacs>=0.1.8",
        "networkx>=2.6.3",
        "joblib>=1.1.0",
    ],

)
