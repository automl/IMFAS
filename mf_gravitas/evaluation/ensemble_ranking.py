import torchsort

from mf_gravitas.losses.ranking_loss import spearman
from mf_gravitas.trainer.rank_ensemble import Trainer_Ensemble


# todo for now take the trainer.evaluation() protocol!  - look into trainer.rank_trainer !
def ranking_eval(model, test_loader):
    """This is a concept function for"""
    trainer_kwargs = {
        'model': model,
        'loss_fn': spearman,
        'ranking_fn': torchsort.soft_rank,
        'optimizer': None,
    }

    # Initialize the trainer
    trainer = Trainer_Ensemble(**trainer_kwargs)

    model.eval()

    trainer.evaluate(test_loader)
    return trainer.losses
