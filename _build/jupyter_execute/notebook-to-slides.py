# Notebook to Slides

## Installation

```
## using conda
!conda install -c conda-forge rise

## using pip
!pip install RISE
!jupyter-nbextension install rise --py --sys-prefix
!jupyter-nbextension enable rise --py --sys-prefix
```

## Use

- Render the notebook into a rj slide:

```
jupyter nbconvert my-nb-slide.ipynb --to slides --post serve
```

- add metadata to Jupyter notebook to set parameters of `reveal.js`

```
{
    ...
    "livereveal": {
        "theme": "serif",
        "transition": "zoom",
        ...
    },
    ...
}
```

- With RISE, we can run a live notebook slide by directly using the notebook file (`alt + r`)