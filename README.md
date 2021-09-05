# GrammarVae_Paper
An implementation of the Grammar VAE Paper and new architecture to improve results


## Training

If data does not exist, run:
* `data\make_innovative_dataset.py`

To train the model:
* `python grammar_vae.py` % the grammar model

attributes:
* `--epochs` % number of epochs
* `--batch_size` % batch size
* `--latent_dim` % size of latent space vector
* `--data` % data set name
* `--model_name` % model name 
* `--data_type` % model name

Example:
* `python grammar_vae.py --latent_dim=56 --epochs=50` % train a model with a 56D latent space and 50 epochs


To compute the model results:

Go to Theano-master and run:
* `python setup.py install`

The experiments with molecules require the rdkit library, which can be installed as described in http://www.rdkit.org/docs/Install.html.

1 - To generate the latent representations, go to:

`equation_optimization/latent_features_and_targets_grammar/` or `molecule_optimization/latent_features_and_targets_grammar/`

and run:
* `python generate_latent_features_and_targets.py`

2 - To run all simulations go to:
`equation_optimization/` or `molecule_optimization/`
and run:
* `python run_all_simulations.py`

3 - Extract the final results by going to:

`equation_optimization/` or `molecule_optimization/`

and run:
* `python get_final_results.py`
