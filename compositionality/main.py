import argparse
import os

import torch
from torch.utils.data import DataLoader
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality.games import PretrainingmGameGS, CompositionalGameGS
from compositionality.wrappers import RnnReceiverGS
from compositionality.callbacks import NeptuneMonitor, EarlyStopperAccuracy, CompositionalityMetricGS
from compositionality.agents import Receiver, Sender
from visual_data.visual_data import prepare_datasets
from visual_data.pretrain import Vision


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')
    parser.add_argument('--rnn_cell', type=str, default='rnn')
    parser.add_argument('--pretraining_sender_lr', type=float, default=1e-3,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--pretraining_receiver_lr', type=float, default=1e-3,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")
    parser.add_argument('--sender_lr', type=float, default=1e-3,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--receiver_lr', type=float, default=1e-5,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")

    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--pretrain', action='store_true', default=False, help="")
    parser.add_argument('--padding', action='store_true', default=True, help="")
    parser.add_argument('--config', type=str, default=None)

    args = core.init(parser)
    print(args)
    return args


if __name__ == "__main__":
    opts = get_params()
    opts.on_slurm = os.environ.get('SLURM_JOB_NAME', False)
    core.util._set_seed(opts.seed)
    full_dataset, train, test = prepare_datasets()
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=opts.batch_size, drop_last=False, shuffle=False)
    vision_module = Vision.from_pretrained('visual_data/vision_model.pth')
    pretrained_senders = [
        core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, vision_module),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=1,
            temperature=3.,
            trainable_temperature=True,
            cell=opts.rnn_cell,
            force_eos=False
        )
        for i in range(2)]
    sender_3 = core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, vision_module),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            temperature=3.,
            trainable_temperature=True,
            force_eos=False,
            cell=opts.rnn_cell)
    receiver = RnnReceiverGS(
            agent=Receiver(opts.receiver_hidden, opts.n_features),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell)

    neptune.init('anonymous/anonymous', backend=neptune.OfflineBackend())
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['']) as experiment:

        # Pretraining game
        if opts.pretrain:
            pretraining_game = PretrainingmGameGS(pretrained_senders, receiver, padding=opts.padding)
            sender_params = [{'params': sender.parameters(), 'lr': opts.pretraining_sender_lr}
                             for sender in pretrained_senders]
            receiver_params = [{'params': receiver.parameters(), 'lr': opts.pretraining_receiver_lr}]
            optimizer = torch.optim.Adam(sender_params + receiver_params)
            trainer = core.Trainer(
                game=pretraining_game, optimizer=optimizer, train_data=train_loader,
                validation_data=test_loader,
                callbacks=[
                    CompositionalityMetricGS(full_dataset, pretrained_senders[0], opts, opts.vocab_size, prefix='1_'),
                    CompositionalityMetricGS(full_dataset, pretrained_senders[1], opts, opts.vocab_size, prefix='2_'),
                    NeptuneMonitor(prefix='pretrain'),
                    core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                    EarlyStopperAccuracy(threshold=0.95, field_name='accuracy', delay=1, train=False),
                ])
            trainer.train(n_epochs=500_000)
            pretraining_game.train(False)

        # Compositional game
        compositional_game = CompositionalGameGS(sender_3, receiver)
        sender_params = [{'params': sender_3.parameters(), 'lr': opts.sender_lr}]
        receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr}]
        optimizer = torch.optim.Adam(sender_params + receiver_params)
        trainer = core.Trainer(game=compositional_game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetricGS(full_dataset, sender_3, opts, opts.vocab_size, prefix='comp'),
                                   NeptuneMonitor(prefix='comp'),
                                   core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                                   EarlyStopperAccuracy(threshold=0.99, field_name='accuracy', delay=10, train=True),
                               ])
        trainer.train(n_epochs=opts.n_epochs)
