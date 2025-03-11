import os
import numpy as np
import typer
import json
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from translast.modeling.albert import AlbertConfig, AlbertModel
from translast.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

class TrainerConfig:
    def __init__(self, seed: int = 3469, 
                 batch_size: int = 128,
                 mini_batch_size: int = 128,
                 num_train_batches: int = 32,
                 num_eval_batches: int = 4,
                 learning_rate: float = 5e-5,
                 n_epochs: int = 25,
                 warmup: float = 0.1, 
                 save_steps: int = 100,
                 total_steps: int = 1000,
                 save_frequency: int = 500,
                 data_parallel: bool = False):
        
        self.seed = seed
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.num_train_batches = num_train_batches
        self.num_eval_batches = num_eval_batches
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.warmup = warmup
        self.save_steps = save_steps
        self.total_steps = total_steps
        self.save_frequency = save_frequency
        self.data_parallel = data_parallel
    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class Trainer:
    """
    Training Helper Class

    This class provides helper functions to train and evaluate a PyTorch model. It handles the training loop, evaluation loop,
    model saving, and loading. The optimizer used is `torch.optim.Adam`.

    Args:
        config (TrainerConfig): Configuration object containing training parameters.
        model (nn.Module): The PyTorch model to be trained.
        data_iter (iter): Iterator to load data.
        device (str): Device name (e.g., 'cpu' or 'cuda').

    Methods:
        train(loss_function, model_file=None, data_parallel=False):
            Runs the training loop.

        eval(evaluate, model_file, data_parallel=True):
            Runs the evaluation loop.

        load(model_file):
            Loads a saved model or pretrained transformer.

        save(i):
            Saves the current model state.
    """

    def __init__(self, config, model, data_iter, device):
        self.config = config
        self.model = model
        self.data_iter = data_iter
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.device = device

    def train(self, loss_function, model_file=None, data_parallel=False):
        """
        Train Loop

        This method runs the training loop for the model. It handles loading the model, training, saving checkpoints,
        and handling data parallelism.

        Args:
            loss_function (function): Function to calculate the loss.
            model_file (str, optional): Path to a saved model file to load. Defaults to None.
            data_parallel (bool, optional): Whether to use Data Parallelism with Multi-GPU. Defaults to False.
        """
        self.model.train()  # train mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:  # If Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0  # global iteration steps regardless of epochs
        for e in range(self.config.n_epochs):
            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = loss_function(model, batch, global_step).mean()  # Average of the Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                if global_step % self.config.save_steps == 0:  # save
                    self.save(global_step)

                if self.config.total_steps and self.config.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.config.n_epochs, loss_sum / (i + 1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step)  # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f' % (e + 1, self.config.n_epochs, loss_sum / (i + 1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """
        Evaluation Loop

        This method runs the evaluation loop for the model. It handles loading the model, evaluating, and handling data parallelism.

        Args:
            evaluate (function): Function to evaluate the model.
            model_file (str): Path to a saved model file to load.
            data_parallel (bool, optional): Whether to use Data Parallelism with Multi-GPU. Defaults to True.

        Returns:
            list: A list of prediction results.
        """
        self.model.eval()  # evaluation mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = []  # prediction results
        iter_tqdm = tqdm(self.data_iter, desc='Iteraction (loss=X.XXX)')
        for batch in iter_tqdm:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # Not calule the gradient
                accuracy, result = evaluate(model, batch)
            results.append(result)
            iter_tqdm.set_description('Iteraction (acc=%5.3f)' % accuracy)
        return results

    def load(self, model_file):
        """
        Load saved model or pretrained transformer (a part of model)

        Args:
            model_file (str): Path to the saved model file.
        """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

    def save(self, i):
        """
        Save current model

        Args:
            i (int): The current global step.
        """
        if os.path.isdir(self.config.output_dir) and self.config.do_train and not self.config.overwrite_output_dir:
            torch.save(self.config.output_dir,  # save model object before nn.DataParallel
                       os.path.join(self.save_dir, 'model_steps_' + str(i) + '.pt'))
            
class EpochMetric:
    def __init__(self):
        self.loss = 0.0

        self.cls_loss = 0.0
        self.cls_accuracy = 0.0

        self.token_loss = 0.0
        self.token_accuracy = 0.0

        self.nb_updates = 0

    def update(self, loss, cls, token):
        self.loss += loss

        self.cls_loss += cls[0]
        self.cls_accuracy += cls[1]

        self.token_loss += token[0]
        self.token_accuracy += token[1]

        self.nb_updates += 1

    def __str__(self):
        s = ""
        s += f"loss: {self.loss / self.nb_updates} "
        s += f"| cls: [loss: {self.cls_loss / self.nb_updates}, accuracy: {100.0 * self.cls_accuracy / self.nb_updates:.2f}%] "
        s += f"| token: [loss: {self.token_loss / self.nb_updates}, accuracy: {100.0 * self.token_accuracy / self.nb_updates:.2f}%]"
        return s

def train_epoch(trainer, dataset, batch_processor, config, args):
    epoch_metrics = EpochMetric()
    update_frequency = config.batch_size // config.mini_batch_size
    pb = tqdm.tqdm(range(config.num_train_batches), disable=~args.tqdm)
    
    for _ in pb:
        idx = np.random.choice(len(dataset['train']), config.batch_size)
        
        for i in range(update_frequency):
            batch = dataset['train'][idx[i*config.mini_batch_size:(i+1)*config.mini_batch_size]]
            batch = batch_processor(batch)
            metrics = trainer.train_step(batch)
            epoch_metrics.update(*metrics)
        
        if not args.no_save and (trainer.ts + 1) % config.save_frequency == 0:
            trainer.save_checkpoint(args.chk_dir / f"albert-{args.model}-checkpoint-{str(trainer.ts).zfill(7)}.pt")
        
        trainer.ts += 1
        display = f"training | ts: {str(trainer.ts).zfill(7)} | {str(epoch_metrics)}"
        pb.set_description(display)
        pb.update(1)
    
    if not args.tqdm:
        logger.info(display)
    
    return epoch_metrics

def evaluate_epoch(trainer, dataset, batch_processor, config, args):
    epoch_metrics = EpochMetric()
    pb = tqdm.tqdm(range(config.num_eval_batches * (config.batch_size // config.mini_batch_size)), disable=~args.tqdm)
    
    for _ in pb:
        idx = np.random.choice(len(dataset['test']), config.mini_batch_size)
        batch = batch_processor(dataset['test'][idx])
        metrics = trainer.eval_step(batch)
        epoch_metrics.update(*metrics)
        
        display = f"evaluation | ts: {str(trainer.ts).zfill(7)} | {str(epoch_metrics)}"
        pb.set_description(display)
        pb.update(1)
    
    if not args.tqdm:
        logger.info(display)
    
    return epoch_metrics

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # logger.info("Training ALBERT model...")
    # trainer_config = TrainerConfig.from_json()
    # albert_config = AlbertConfig.from_json()
    # model = AlbertModel(albert_config)
    # data_iter = ...  # Replace with your data iterator
    # save_dir = "models"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trainer = Trainer(trainer_config, model, data_iter, save_dir, device)
    # trainer.train(loss_function, data_parallel=True)

    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # # -----------------------------------------
    pass

if __name__ == "__main__":
    app()
