import h5py
import numpy as np
import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Network import GrammarVariationalAutoEncoder, VAELoss
from grammar import mol_grammar as G

class ModelManager():
    def __init__(self, model, train_step_init=0, lr=1e-3):
        self.train_step = train_step_init
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = VAELoss()

    def train(self, loader):
        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        _losses = []
        for batch_idx, data in enumerate(loader):
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            data = Variable(data)
            data = data.to(device)

            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_fn(data, mu, log_var, recon_batch)
            loss_value = loss.data.numpy()
            _losses.append(loss_value)
            loss.backward()
            self.optimizer.step()
            self.train_step += 1

            batch_size = len(data)

            if batch_idx == 0:
                print(f'batch size: {batch_size}')
            if batch_idx % 50 == 0:
                print(f'training loss: {loss_value}')

        return _losses

    def test(self, loader):

        self.model.eval()
        test_loss = 0
        for batch_idx, data in enumerate(loader):
            with torch.no_grad():
                data = Variable(data)
                data = data.to(device)

                recon_batch, mu, log_var = self.model(data)
                loss = self.loss_fn(data, mu, log_var, recon_batch)
                test_loss += (loss * len(data))

        print(f'testset length is: {len(loader.dataset)}')

        test_loss /= len(loader.dataset)
        print(f'====> Test set loss: {test_loss}')


def kfold_loader(data):

    with h5py.File(data, 'r') as h5f:
        if 'zinc' in data or 'qm9' in data:
            data_as_array = h5f['data'][:]
            np.random.shuffle(data_as_array)
            train = data_as_array[25000::]
            test = data_as_array[:25000:]
            return torch.FloatTensor(train), torch.FloatTensor(test)

        train = np.concatenate([h5f['data'][i::10] for i in range(1, 10)])
        test = np.concatenate([h5f['data'][i::10] for i in range(0, 1)])

        return torch.FloatTensor(train), torch.FloatTensor(test)


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Molecular autoencoder network')
    parser.add_argument('--model_name', type=str, metavar='N', default="mol_grammar_selfie")
    parser.add_argument('--epochs', type=int, metavar='N', default=50,
                        help='Number of epochs to run during training.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=700,
                        help='batch size')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=56,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--data', type=str, metavar='N',
                        default="data/zinc_selfie_grammar_dataset.h5",
                        help='data set file')
    parser.add_argument('--data_type', type=str, metavar='N',
                        default="zinc",
                        help='data set name')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()

    epochs = args.epochs
    batch_size = args.batch_size
    data_type = args.data_type
    model_name = args.model_name

    data = args.data
    rules = G.gram.split('\n')

    train, test = kfold_loader(data)

    train_loader = torch.utils.data \
        .DataLoader(train, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils \
        .data.DataLoader(test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    losses = []
    vae = GrammarVariationalAutoEncoder(model_name=model_name, rules=rules)
    vae.to(device)

    model_mgr = ModelManager(vae, lr=2e-3)
    for epoch in range(1, epochs + 1):
        losses += model_mgr.train(train_loader)
        print('epoch {} complete'.format(epoch))
        model_mgr.test(test_loader)

    filepath = f'final_models/{data_type}_selfie_{model_name}_dataset_L' + \
        str(args.latent_dim) + '_E' + str(epochs) + '_val.hdf5'
    torch.save(model_mgr.model.state_dict(), filepath)