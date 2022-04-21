import torch.nn as nn

from mf_gravitas.trainer.base_trainer import train


def _freeze(listoflayers, unfreeze=True):
    """freeze the parameters of the list of layers """
    for l in listoflayers:
        for p in l.parameters():
            p.requires_grad = unfreeze


def freeze_train(model, epochs, *args, **kwargs):
    """
    unfreeze from the outer to the inner of the autoencoder
    :param epochs: total epochs that are to be trained
    :param args: args passed to _train()
    """
    # find those layers that need to be frozen
    enc = [l for l in model.encoder if isinstance(l, nn.Linear)]
    dec = [l for l in reversed(model.decoder) if isinstance(l, nn.Linear)]

    enc_bn = [l for l in model.encoder if isinstance(l, nn.BatchNorm1d)]
    dec_bn = [l for l in reversed(model.encoder) if isinstance(l, nn.BatchNorm1d)]

    # linear intervals for unfreezing
    no_freeze_steps = len(enc)

    # freeze all intermediate layers
    _freeze([*enc, *dec, *enc_bn, *dec_bn], unfreeze=False)

    # scheduler for the layers that are iteratively unfrozen
    unfreezer = zip(enc, dec, enc_bn, dec_bn)

    for i in range(no_freeze_steps):
        # determine which layers are melted
        en, de, ebn, dbn = unfreezer.__next__()
        _freeze([en, de, ebn, dbn], unfreeze=True)
        train(model, epochs=epochs // no_freeze_steps, *args, **kwargs)
