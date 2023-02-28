# UPerNet
This repo runs Huggingface's implementation of [UPerNet](https://huggingface.co/docs/transformers/model_doc/upernet).

The code is borrowed from this [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UPerNet).

# Run with Moreh framrework
        conda create -n upernet python=3.8
        conda activate upernet
        update-moreh --force --targer 23.3.0

# Inference
        python upernet.py