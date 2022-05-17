# SSR-TA

This is the code for the "SSR-TA: Sequence to Sequence based expert recurrent recommendation for ticket automation" and baselines.

pygcn https://github.com/tkipf/pygcn was used to implement the baseline DeepRouting.

Pre-request

> linux 16.04
> python 3.8.8
> PyTorch 1.9.1
> CUDA Version: 10.2

Usage

- Clone the repo: https://github.com/ismango/SSR-TA.git
- Prepare your train/Val/test data and preprocess the data.
- Refer to the codes of corresponding sections for specific purposes.


Model
- load_data.py
> The code to load the ticket dataset.

- model.py
> The codes of the models in this paper.

Train&&Test
- train.py
> The codes to train the SSR-TA and evaluate the network on the test data.
