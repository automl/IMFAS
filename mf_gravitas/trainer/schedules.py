import logging

from mf_gravitas.trainer.autoencoder import Trainer_Autoencoder

# A logger for this file
log = logging.getLogger(__name__)


# TODO: consider how to refactor these (since they are very much structured - and should
#  be configurable


def train_schedule(model, train_dataloader, test_dataloader, epochs=[100, 100, 100], lr=0.001):
    # Consider Marius idea to first find a reasonable data representation
    #  and only than train with the algorithms

    train_args = {
        'model': model,
        'loss_fn': None,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'lr': lr,
        'epochs': None

    }

    trainer = Trainer_Autoencoder()

    # pretrain
    name = model.__class__.__name__
    log.info(f"\nPretraining {name} with reconstruction loss:")
    train_args['loss_fn'] = model._loss_reconstruction
    train_args['epochs'] = epochs[0]
    trainer.train(**train_args)

    # train datasets
    log.info(f"\nTraining {name} with dataset loss:")
    train_args['loss_fn'] = model._loss_datasets
    train_args['epochs'] = epochs[1]
    trainer.train(**train_args)

    # train algorithms
    log.info(f"\nTraining {name} with algorithm:")
    train_args['loss_fn'] = model._loss_algorithms
    train_args['epochs'] = epochs[2]
    trainer.train(model, model._loss_algorithms, train_dataloader, test_dataloader, epochs[2], lr)


def train_gravity(model, train_dataloader, test_dataloader, epochs, lr=0.001):
    """
    Two step training:
    1) Pretraining using reconstruction loss
    2) Training using gravity loss (reconstruction + attraction + repellent +
    :param train_dataloader:
    :param test_dataloader:
    :param epochs: list of int (len 2): epochs for step 1) and 2) respectively
    :param lr:
    :return:
    """

    train_args = {
        'model': model,
        'loss_fn': None,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'lr': lr,
        'epochs': None

    }

    trainer = Trainer_Autoencoder()

    name = model.__class__.__name__
    log.info(f"\nPretraining {name} with reconstruction loss: ")
    train_args['loss_fn'] = model._loss_reconstruction
    train_args['epochs'] = epochs[0]
    trainer.train(**train_args)

    log.info(f"\nTraining {name} with gravity loss:")
    train_args['loss_fn'] = model.loss_gravity
    train_args['epochs'] = epochs[1]
    trainer.train(**train_args)


def train_freeze_schedule(model, train_dataloader, test_dataloader, epochs=[100, 100, 100],
                          lr=0.001):
    # Consider Marius idea to first find a reasonable data representation
    #  and only than train with the algorithms

    train_args = {
        'model': model,
        'loss_fn': None,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'lr': lr,
        'epochs': None

    }

    trainer = Trainer_Autoencoder()

    # pretrain
    name = model.__class__.__name__
    log.info(f'\nPretraining {name} with reconstruction loss:')
    train_args['loss_fn'] = model._loss_reconstruction
    train_args['epochs'] = epochs[0]
    trainer.freeze_train(**train_args)

    # train datasets
    log.info(f'\nTraining {name} with dataset loss:')
    # model._train(model._loss_datasets, train_dataloader, test_dataloader, epochs=epochs[1], lr=lr)
    train_args['loss_fn'] = model._loss_datasets
    train_args['epochs'] = epochs[1]
    trainer.freeze_train(**train_args)

    # train algorithms
    log.info(f'\nTraining {name} with algorithm:')
    train_args['loss_fn'] = model._loss_algorithms
    train_args['epochs'] = epochs[2]
    trainer.train(**train_args)
