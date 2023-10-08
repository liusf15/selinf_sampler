# selinf_sampler

Code for conducting post-selection inference after lasso selection.

The algorithm is proposed in Liu, S. (2023). [An Exact Sampler for Inference after Polyhedral Model Selection.](https://arxiv.org/abs/2308.10346)

#### Installation:
```
git clone https://github.com/liusf15/selinf_sampler.git
```

The proposed SOV sampler is written in cython so it requires compiling:
```
cd src
python setup.py build_ext --inplace
```

The [vignette](examples/vignette.ipynb) provides an example showing how to use the implemented method to construct confidence intervals and compute p-values for variables selected by the lasso. The lasso uses a subset of the data and the inference is performed conditioned on the selection.
