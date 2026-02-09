### TODO
- [ ] rec_loss -> nll_loss rationalization (Kendall-Gal, 2018)


### CMD

```sh
torchrun --nproc-per-node 2 run/train_clear_camelyon.py --config camelyon --experiment-name test01 --run-name test-ddp
```

### [MNIST-C](https://github.com/google-research/mnist-c) module `corrpution_utils`

Check https://docs.wand-py.org/en/0.6.7/guide/install.html for installation of `ImageMagick`.
For Ubuntu operating system (e.g. `Google Colab`), try this:
```
apt-get install libmagickwand-dev
pip install wand
```