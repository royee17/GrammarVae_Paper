# GrammarVae_Paper
An implementation of the Grammar VAE Paper and new architecture to improve results


## Training

### Molecules

To train the model:

* `python grammar_vae.py` % the grammar model

attributes:
* `--epochs` % number of epochs
* `--batch_size` % batch size
* `--latent_dim` % size of latent space vector
* `--data` % data set name
* `--model_name` % model name

Example:
* `python grammar_vae.py --latent_dim=56 --epochs=50` % train a model with a 56D latent space and 50 epochs 

To compute the model results:

1 - To generate the latent representations, go to:

`equation_optimization/latent_features_and_targets_grammar/`

and run:
* `python generate_latent_features_and_targets.py`

2 - For each of the simulation directories (1-10) run:
* `python run_bo.py`

3 - Extract the final results by going to:

`equation_optimization/`

and run:
* `python get_final_results.py`

and:
* `./get_average_test_RMSE_LL.sh`
