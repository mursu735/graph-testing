## Prerequisites for running spaCy with en_core_web_trf model

1. Install torch with CUDA enabled, instructions are available on PyTorch website
 (if torch is already installed, it should be uninstalled first)
2. Run `pip install -r requirements.txt` this should download all models

If this does not work, replace model with `en_core_web_sm`


## How to install Graphviz and Pygraphviz on Windows

1. Download and install Graphviz v2.46.0 https://gitlab.com/graphviz/graphviz/-/package_files/6164164/download (later versions may or may not work)
2. Run `python -m pip install --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz`
(replace path to Graphviz directories if needed)