from copy import deepcopy

class EarlyStopping:
    """
    Simple class for doing early-stopping.
    """
    def __init__(self, patience=None, min_epochs=None):
        self.patience = patience
        self.min_epochs = min_epochs

        self.best_loss = float('inf')
        self.best_epoch = None
        self.best_step = None
        self.best_model = None
        self.no_improvement_count = 0
        self.stop = False

        # Don't do early stopping if patience or min_epochs is none
        if self.patience is None or self.min_epochs is None:
            self.do_early_stopping = False
        else:
            self.do_early_stopping = True


    def check(self, model, loss, epoch, step):
        """
        Checks if training should be stopped.
        """
        if not self.do_early_stopping:
            return False

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_step = step
            self.best_model = deepcopy(model.state_dict())
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if (
            self.no_improvement_count >= self.patience
            and epoch > self.min_epochs - 1
        ):
            self.stop = True

        return self.stop
