__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

from pathlib import Path

import torch
import math
import sys
import os
import wandb
import matplotlib

import matplotlib.pyplot as plt
# from dacite.types import is_instance


matplotlib.use('Agg')

from typing import List, Optional, Union, Callable, Any

from ..utils.logger import CustomLogger
from ..eval.evaluators import Evaluator
from .adaloss import Adaloss

from ..utils.registry import Registry

TRAINERS = Registry('trainers', module_key='animaloc.train.trainers')

__all__ = ['TRAINERS', *TRAINERS.registry_names]

@TRAINERS.register()
class Trainer:
    ''' Base class for training a model '''

    def __init__(
        self, 
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int, 
        lr_milestones: Optional[List[int]] = None,  
        auto_lr: Union[bool, dict] = False,
        adaloss: Optional[str] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_loss_dataloader: Optional[torch.utils.data.DataLoader] = None,
        evaluator: Optional[Evaluator] = None,
        vizual_fn: Optional[Callable] = None,
        debug_vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
        device_name: str = 'cuda', 
        print_freq: int = 50,
        valid_freq: int = 1,
        csv_logger: bool = False,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        wandb_artifact_upload: bool = False,

        ) -> None:
        '''
        Args:
            model (torch.nn.Module): CNN model to train, that takes as 
                inputs image and target and returns a dict loss for training, and both output
                and dict loss for evaluation.
            train_dataloader (torch.utils.data.DataLoader): a pytorch's DataLoader used for 
                model training that combines a dataset and a sampler, and provides an iterable 
                over the given dataset. 
                see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            optimizer (torch.optim.Optimizer): a pytorch's optimization algorithm.
                see https://pytorch.org/docs/stable/optim.html
            num_epochs (int): number of epochs for training the model
            lr_milestones (list, optional): learning rate (lr) milestones representing a list
                of epoch indices (must be increasing) to build a lr scheduler that decays 
                the learning rate of each parameter group by 0.1 once the number of epoch 
                reaches one of the milestones. 
                Defaults to None.
            auto_lr (bool, optional): set to True to use the Pytorch's ReduceLROnPlateau scheduler
                with default parameters values. If specified, cancels the use of LR milestones 
                (i.e. MultiStepLR). If a dict is specified, it must contain some of the scheduler's 
                parameters with the parameter name as key, and the associated value.
                Defaults to False.
            adaloss (str, optional): specify the name of a dataset's end-transform parameter to be
                updated during training. This use "Adaloss", an objective function that adapts itself
                during the training by updating a parameter based on the training statistics (see
                https://arxiv.org/abs/1908.01070).
                Note that the dataset must contain an "update_end_transform()" method and the parameter
                chosen must be stored in a "end_params" attribute.
                Defaults to None.
            val_dataloader (torch.utils.data.DataLoader, optional): a pytorch's DataLoader used for 
                model validation that combines a dataset and a sampler, and provides an iterable 
                over the given dataset. 
                Defaults to None.
            evaluator (Evaluator, optional): if specified, used for evaluation.
                Override the default evaluator.
                Defaults to None.
            vizual_fn (callable, optional): a model specific function that will be use for plotting
                samples in Weights & Biases during validation. It must take 'image', 'target', and 
                'output' as arguments and return a matplotlib figure. Defaults to None.
            work_dir (str, optional): directory where checkpoints and logs will be saved. If None
                is given, results and logs files will be saved in current working directory.
                Defaults to None.
            device (str): the device name on which tensors will be allocated ('cpu' or 'cuda'). 
                Defaults to 'cuda'.
            print_freq (int, optional): define the frequency at which the logs will be
                printed and/or recorded. 
                Defaults to 50.
            valid_freq (int, optional): define the frequency at which the model will be validated.
                Note that first and last epoch are always validated.
                Defaults to 1 (i.e., after each epoch).
            csv_logger (bool, optional): set to True to store logs in a CSV file. Warning, long
                training session might slow down the process.
                Defaults to False.
        '''

        assert isinstance(model, torch.nn.Module), \
            f'model argument must be an instance of nn.Module(), ' \
                f'got \'{type(model)}\''
        
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), \
            f'train_dataloader argument must be an instance of ' \
                f'torch.utils.data.DataLoader(), got \'{type(train_dataloader)}\''
        
        assert isinstance(val_dataloader, (torch.utils.data.DataLoader, type(None))), \
            f'val_dataloader argument must be an instance of ' \
                f'torch.utils.data.DataLoader(), got \'{type(val_dataloader)}\''
        
        assert isinstance(optimizer, torch.optim.Optimizer), \
            f'optimizer argument must be an instance of ' \
                f'torch.optim.Optimizer(), got \'{type(optimizer)}\''
        
        assert isinstance(lr_milestones, (list, type(None))), \
            f'lr_milestones argument must be a list, got \'{type(lr_milestones)}\''
        
        assert isinstance(auto_lr, (bool, dict)), \
            f'auto_lr argument must be a bool or a dict, got \'{type(auto_lr)}\''

        assert isinstance(evaluator, (Evaluator, type(None))), \
            f'evaluator argument must be an instance of Evaluator class, ' \
                f'got \'{type(evaluator)}\''
        
        assert callable(vizual_fn) or isinstance(vizual_fn, type(None)), \
            f'vizual_fn argument must be a callable function, got \'{type(vizual_fn)}\''
        
        assert valid_freq <= num_epochs, \
            'validation frequency must be lower or equal to the number of epochs'
        
        self.device = torch.device(device_name)

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.val_loss_dataloader = val_loss_dataloader
        self.optimizer = optimizer
        self.epochs = num_epochs
        
        self.print_freq = print_freq
        self.valid_freq = valid_freq
        self.lr_milestones = lr_milestones
        self.evaluator = evaluator

        self.vizual_fn = vizual_fn

        # auto-learning rate reduction
        self.auto_lr = auto_lr
        self.auto_lr_flag = False
        if auto_lr or isinstance(auto_lr, dict):
            self.auto_lr_flag = True

        self.wandb_artifact_upload = wandb_artifact_upload
        # adaloss
        self.adaloss = adaloss
        if isinstance(adaloss, str):
            assert 'end_params' in dir(self.train_dataloader.dataset), \
                'end_params attribute is missing from the training dataset'
            assert 'update_end_transforms' in dir(self.train_dataloader.dataset), \
                'update_end_transforms method is missing from the training dataset'
            assert adaloss in self.train_dataloader.dataset.end_params.keys(), \
                'Adaloss specified parameter is missing from the training dataset ' \
                    'end-transforms parameters'
            
            self.adaparam = adaloss
            self.adaloss = Adaloss(
                train_dataloader.dataset.end_params[adaloss], w=3, delta_max=1)

        # working directory
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = os.getcwd()

        # loggers
        self.csv_logger = csv_logger
        self.train_logger = CustomLogger(delimiter=' ', filename='training', work_dir=self.work_dir, csv=self.csv_logger)
        self.val_logger = CustomLogger(delimiter=' ', filename='validation', work_dir=self.work_dir, csv=self.csv_logger)

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        # Early stopping tracking variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def prepare_data(self, images, targets) -> tuple:
        ''' Method to prepare the data before feeding to the model. 
        Can be override by subclass to create a custom Trainer.

        Args:
            images,
            targets
        
        Returns:
            tuple
        '''

        images = images.to(self.device)

        if isinstance(targets, (list, tuple)):
            targets = [tar.to(self.device) for tar in targets]
        else:
            targets = targets.to(self.device)

        return images, targets

    def start(
        self, 
        warmup_iters: Optional[int] = None, 
        checkpoints: str = 'best',
        select: str = 'min',
        validate_on: str = 'all',
        wandb_flag: bool = False
        ) -> torch.nn.Module:
        ''' Start training from epoch 1 
        
        Args:
            warmup_iters (int, optional): number of iterations to warm up the
                training, i.e. gradually increase the learning rate to reach its
                initial specified value.
            checkpoints (str, optional): mode for saving the checkpoints. Possible 
                values are:
                    - 'best' (default), to save the best checkpoint (based on validation
                        output),
                    - 'all', to save all the checkpoints.
                If no validation, all checkpoints are saved.
                Defaults to 'best'.
            select (str, optional): best epoch selection mode, used if 'checkpoints' is set
                to best. Possible values are:
                    - 'min' (default), for selecting the epoch that yields to a minimum validation value,
                    - 'max', for selecting the epoch that yields to a maximum validation value. 
                Defaults to 'min'.
            validate_on (str, optional): metrics/loss used for validation (i.e. best model and auto-lr).
                For validation with losses, possible values are the names returned by the model, or 'all'
                for using the sum of all losses (default). Possible values for evaluator are: 'recall', 
                'precision', 'f1_score', 'mse', 'mae', 'rmse', 'accuracy' or 'mAP'. 
                Defauts to 'all'
            wandb_flag (bool, optional): set to True to log on Weight & Biases. Defaults to False.
        
        Returns:
            torch.nn.Module:
                trained model
        '''

        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        lr_scheduler = self._lr_scheduler()
        val_flag = False

        if select =='min': 
            self.best_val = float('inf')
        elif select =='max': 
            self.best_val = 0

        self.best_val_loss = float('inf')

        # Reset early stopping variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if wandb_flag:
            wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

            learning_rates = len(self.optimizer.param_groups)
            for i in range(1, learning_rates):
                wandb.log({f'lr_{i}': self.optimizer.param_groups[i]["lr"]})

        for epoch in range(1, self.epochs + 1):

            # training
            train_output = self._train(epoch, warmup_iters, wandb_flag)
            if wandb_flag:
                wandb.log({'train_loss': train_output, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]["lr"], 'epoch': epoch})

                learning_rates = len(self.optimizer.param_groups)
                for i in range(1, learning_rates):
                    wandb.log({f'lr_{i}': self.optimizer.param_groups[i]["lr"]})

            # validation
            if epoch % self.valid_freq == 0 or epoch in [1, self.epochs]:
                val_output = None
                val_loss_output = None
                val_output_total_loss = None

                if self.evaluator is not None:
                    val_flag = True
                    viz = False
                    if wandb_flag: viz = True
                    self._prepare_evaluator('validation', epoch)
                    val_output = self.evaluator.evaluate(returns=validate_on, viz=viz, wandb_flag=False)

                    logger.info(f'{self.evaluator.header} {validate_on}: {val_output:.4f}')

                    if wandb_flag:
                        wandb.log({
                            validate_on: val_output,
                            'epoch': epoch,
                            'f1_score': self.evaluator.metrics.fbeta_score(c=1, beta=1),
                            'f2_score': self.evaluator.metrics.fbeta_score(c=1, beta=2),
                            'f5_score': self.evaluator.metrics.fbeta_score(c=1, beta=5),
                            'true_positive': sum(self.evaluator.metrics.tp),
                            'false_negative': sum(self.evaluator.metrics.fn),
                            'false_positive': sum(self.evaluator.metrics.fp),
                            'n': sum(self.evaluator.metrics.tp) + sum(self.evaluator.metrics.fn) + sum(
                                self.evaluator.metrics.fp),
                            'recall': self.evaluator.metrics.recall(),
                            'precision': self.evaluator.metrics.precision(),
                            'mse': self.evaluator.metrics.mse(),
                            'mae': self.evaluator.metrics.mae(),
                            'me': self.evaluator.metrics.me(),
                            'rmse': self.evaluator.metrics.rmse(),
                            'accuracy': self.evaluator.metrics.accuracy(),
                            'avg_scores': self.evaluator.metrics.avg_score(),
                            'avg_dscores': self.evaluator.metrics.avg_dscore(),
                        })
                        # if isinstance(self.evaluator.metrics, DensityAwarePointsMetrics):
                        from collections import defaultdict

                        # Define density buckets
                        def get_bucket(d):
                            """Map density to bucket name."""
                            if isinstance(d, str):
                                # Already a bucket like '15+'
                                if d == '15+':
                                    return '15+'
                                try:
                                    d = int(d)
                                except ValueError:
                                    return None

                            if d == 1:
                                return '1'
                            elif 2 <= d <= 5:
                                return '2_to_5'
                            elif 6 <= d <= 15:
                                return '6_to_15'
                            elif d > 15:
                                return '15+'
                            return None

                        # Aggregate stats per bucket
                        bucket_stats = {name: defaultdict(float) for name in ['1', '2_to_5', '6_to_15', '15+']}

                        for gt_density, density_stats in self.evaluator.metrics.density_stats.items():
                            bucket_name = get_bucket(gt_density)
                            if bucket_name and bucket_name in bucket_stats:
                                bucket_stats[bucket_name]['tp'] += density_stats.tp
                                bucket_stats[bucket_name]['fn'] += density_stats.fn
                                bucket_stats[bucket_name]['fp'] += density_stats.fp
                                bucket_stats[bucket_name]['sum_signed_error'] += density_stats.sum_signed_error
                                bucket_stats[bucket_name]['total_pred'] += density_stats.total_pred
                                bucket_stats[bucket_name]['total_gt'] += density_stats.total_gt
                                bucket_stats[bucket_name]['count'] += 1

                        # Compute derived metrics and log
                        density_metrics = {'epoch': epoch}
                        for bucket_name, stats in bucket_stats.items():
                            prefix = f'density/{bucket_name}'
                            tp, fn, fp = stats['tp'], stats['fn'], stats['fp']

                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            f2 = 5 * precision * recall / (4 * precision + recall) if (
                                                                                                  4 * precision + recall) > 0 else 0
                            mae = abs(stats['sum_signed_error']) / stats['count'] if stats['count'] > 0 else 0

                            density_metrics.update({
                                f'{prefix}/tp': tp,
                                f'{prefix}/fn': fn,
                                f'{prefix}/fp': fp,
                                f'{prefix}/recall': recall,
                                f'{prefix}/precision': precision,
                                f'{prefix}/f1': f1,
                                f'{prefix}/f2': f2,
                                f'{prefix}/mae': mae,
                                f'{prefix}/total_pred': stats['total_pred'],
                                f'{prefix}/total_gt': stats['total_gt'],
                            })

                        wandb.log(density_metrics)


                if self.val_loss_dataloader is not None:
                    val_flag = True
                    val_loss_output = self.evaluate(epoch, wandb_flag=wandb_flag, returns="all",
                                                    custom_val_dataloader=self.val_loss_dataloader)
                    if wandb_flag and isinstance(val_loss_output, dict):
                        for key, value in val_loss_output.items():
                            wandb.log({f'val_{key}': value, 'epoch': epoch})
                    val_output_total_loss = val_loss_output["total_loss"]


                # Early stopping check on the evaluate_on output which is not the loss value
                if val_flag and self.early_stopping and self.evaluator is not None:
                    if self._early_stopping_check(val_output, select, epoch):
                        self.stopped_epoch = epoch
                        logger.info(f'Early stopping triggered at epoch {epoch}')
                        break

                logger.info(
                    f"Checking for best model by Evaluator output at epoch {epoch} with validation output: {val_output}")
                # save checkpoint(s), best by Evalutator output
                if val_flag and checkpoints == 'best' and val_output is not None and self._is_best(val_output, mode = select):
                    model_checkpoint_path = self._save_checkpoint(epoch, checkpoints)
                    logger.info(
                        f'Best model by End User Metric {validate_on} saved - Epoch {epoch} - Validation value: {val_output:.6f}, path: {model_checkpoint_path}')
                    if self.wandb_artifact_upload:
                        artifact = wandb.Artifact(name=checkpoints, type="model")
                        artifact.add_file(model_checkpoint_path)  # Add a file
                        try:
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            logger.error(f'Error logging artifact to wandb: {e}')


                    if wandb_flag:
                        best_metrics_dict = {
                            "best_f1_score": self.evaluator.metrics.fbeta_score(c=1, beta=1),
                            "best_f2_score": self.evaluator.metrics.fbeta_score(c=1, beta=2),
                            "best_f5_score": self.evaluator.metrics.fbeta_score(c=1, beta=5),
                            "best_true_positive": sum(self.evaluator.metrics.tp),
                            "best_false_negative": sum(self.evaluator.metrics.fn),
                            "best_false_positive": sum(self.evaluator.metrics.fp),
                            "best_n": sum(self.evaluator.metrics.tp) + sum(self.evaluator.metrics.fn) + sum(self.evaluator.metrics.fp),
                            "best_recall": self.evaluator.metrics.recall(),
                            "best_precision": self.evaluator.metrics.precision(),
                            "best_mse": self.evaluator.metrics.mse(),
                            "best_mae": self.evaluator.metrics.mae(),
                            "best_me": self.evaluator.metrics.me(),
                            "best_rmse": self.evaluator.metrics.rmse(),
                            "best_accuracy": self.evaluator.metrics.accuracy(),
                            "best_avg_scores": self.evaluator.metrics.avg_score(),
                            "best_avg_dscores": self.evaluator.metrics.avg_dscore(),
                            "best_epoch": epoch
                        }
                        wandb.run.summary['best_validation'] = self.best_val
                        wandb.run.summary.update(best_metrics_dict)

                # best by validation loss
                logger.info(
                    f"Checking for best model by validation loss at epoch {epoch} with validation output: {val_output_total_loss}")
                if val_flag and checkpoints == 'best' and val_output_total_loss is not None and self._is_best_loss(val_output_total_loss):

                    model_checkpoint_path = self._save_checkpoint(epoch, mode="best_loss")
                    logger.info(
                        f'Best model by Validation Loss saved - '
                        f'Epoch {epoch} - '
                        f'Validation value: {val_output_total_loss:.6f}, '
                        f'path: {model_checkpoint_path}'
                    )
                    if self.wandb_artifact_upload:
                        artifact = wandb.Artifact(name=checkpoints, type="model")
                        artifact.add_file(model_checkpoint_path)  # Add a file
                        try:
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            logger.error(f'Error logging artifact to wandb: {e}')


                    if wandb_flag and val_loss_output is not None:
                        best_metrics_dict = {
                        "best_total_loss": val_output,
                        "best_epoch": epoch
                        }
                        if isinstance(val_loss_output, dict):
                            best_metrics_dict.update(val_loss_output)
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary.update(best_metrics_dict)


                elif checkpoints == 'all':
                    self._save_checkpoint(epoch, checkpoints)

            
            self._save_checkpoint(epoch, 'latest')

            # scheduler
            if lr_scheduler is not None :
                if self.auto_lr_flag and self.evaluator is not None and val_output is not None:
                    lr_scheduler.step(val_output)
                elif self.auto_lr_flag and val_output_total_loss is not None:
                    lr_scheduler.step(val_output_total_loss)
                else:
                    lr_scheduler.step()
            
            # adaloss
            if self.adaloss is not None:
                self.adaloss.step()
                self.train_dataloader.dataset.load_end_param(self.adaparam, self.adaloss.param)
                logger.info('Adaloss param: {}'.format(self.train_dataloader.dataset.end_params[self.adaparam]))
                self.train_dataloader.dataset.update_end_transforms()
                self.val_dataloader.dataset.end_params = self.train_dataloader.dataset.end_params
                self.val_dataloader.dataset.update_end_transforms()

            # Restore best weights if early stopping was triggered and restore_best_weights is True
            if self.early_stopping and self.restore_best_weights and self.best_weights is not None:
                logger.info('Restoring best model weights')
                self.model.load_state_dict(self.best_weights)
        
        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            if self.stopped_epoch > 0:
                wandb.run.summary['stopped_epoch'] = self.stopped_epoch
            wandb.run.finish()
        
        return self.model

    def _early_stopping_check(self, current_val: float, mode: str, epoch: int) -> bool:
        ''' Check if early stopping criteria is met '''
        logger.info(f"Check if early stopping criteria is met")
        if mode == 'min':
            # For minimization (e.g., loss)
            if current_val < (self.best_val - self.min_delta):
                self.best_val = current_val
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.state_dict().copy()
            else:
                self.wait += 1

        elif mode == 'max':
            # For maximization (e.g., accuracy)
            if current_val > (self.best_val + self.min_delta):
                self.best_val = current_val
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.state_dict().copy()
            else:
                self.wait += 1

        # Check if patience is exceeded
        if self.wait >= self.patience:
            return True

        return False
    
    def resume(
        self, 
        pth_path: str, 
        checkpoints: str = 'best',
        select: str = 'min',
        validate_on: str = 'recall',
        load_optim: bool = False,
        wandb_flag: bool = False
        ) -> torch.nn.Module:
        ''' Resume training from a pth file 
        
        Args:
            pth_path (str): absolute path to the checkpoint (i.e. pth file)
            checkpoints (str, optional): mode for saving the checkpoints. Possible 
                values are:
                    - 'best' (default), to save the best checkpoint (based on validation
                        output),
                    - 'all', to save all the checkpoints.
                If no validation, all checkpoints are saved.
                Defaults to 'best'.
            select (str, optional): best epoch selection mode, used if 'checkpoints' is set
                to best. Possible values are:
                    - 'min' (default), for selecting the epoch that yields to a minimum validation value,
                    - 'max', for selecting the epoch that yields to a maximum validation value. 
                Defaults to 'min'.
            validate_on (str, optional): metrics used for validation (i.e. best model and auto-lr) when 
                custom evaluator is specified. Possible values are: 'recall', 'precision', 'f1_score', 
                'mse', 'mae', 'rmse' and 'mAP'. 
                Defauts to 'recall'
            load_optim (bool, optional): set to True to load the optimizer's state_dict.
                Defaults to False
            wandb_flag (bool, optional): set to True to log on Weight & Biases. Defaults to False.
        
        Returns:
            torch.nn.Module:
                trained model
        '''

        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optim is True:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        resume_epoch = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.best_val = checkpoint['best_val']


        self.best_val = checkpoint['best_val']

        lr_scheduler = self._lr_scheduler()
        val_flag = False

        if wandb_flag:
            wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

            learning_rates = len(self.optimizer.param_groups)
            for i in range(1, learning_rates):
                wandb.log({f'lr_{i}': self.optimizer.param_groups[i]["lr"]})

        for epoch in range(resume_epoch + 1, self.epochs + 1):

            # training
            train_output = self._train(epoch, wandb_flag=wandb_flag) 
            if wandb_flag:
                wandb.log({'train_loss': train_output, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

                learning_rates = len(self.optimizer.param_groups)
                for i in range(1, learning_rates):
                    wandb.log({f'lr_{i}': self.optimizer.param_groups[i]["lr"]})

            # validation
            if epoch % self.valid_freq == 0 or epoch in [1, self.epochs]:

                if self.evaluator is not None:
                    val_flag = True
                    viz = False
                    if wandb_flag: viz = True
                    self._prepare_evaluator('validation', epoch)
                    val_output = self.evaluator.evaluate(returns=validate_on, viz=viz)
                    logger.info(f'{self.evaluator.header} {validate_on}: {val_output:.4f}')

                    if wandb_flag:
                        wandb.log({validate_on: val_output, 'epoch': epoch})

                elif self.val_dataloader is not None:
                    val_flag = True
                    val_output = self.evaluate(epoch, wandb_flag=wandb_flag, returns=validate_on)
                    if wandb_flag:
                        wandb.log({'val_loss': val_output, 'epoch': epoch})
                
                # save checkpoint(s)
                if val_flag and checkpoints =='best' and self._is_best(val_output, mode = select):
                    model_checkpoint_path = self._save_checkpoint(epoch, checkpoints)
                    logger.info('Best model saved - Epoch {} - Validation value: {:.6f}, path: {}'.format(epoch, val_output, model_checkpoint_path))
                    if self.wandb_artifact_upload:
                        artifact = wandb.Artifact(name=checkpoints, type="model")
                        artifact.add_file(model_checkpoint_path)  # Add a file
                        wandb.log_artifact(artifact)
                elif checkpoints == 'all':
                    self._save_checkpoint(epoch, checkpoints)
            
            self._save_checkpoint(epoch, 'latest')

            # scheduler
            if lr_scheduler is not None:
                if self.auto_lr_flag:
                    if 'val_output' in locals():
                        lr_scheduler.step(val_output)
                    else:
                        lr_scheduler.step(self.best_val)
                else:
                    lr_scheduler.step()
            
            # adaloss
            if self.adaloss is not None:
                self.adaloss.step()
                self.train_dataloader.dataset.load_end_param(self.adaparam, self.adaloss.param)
                self.train_dataloader.dataset.update_end_transforms()
                self.val_dataloader.dataset.end_params = self.train_dataloader.dataset.end_params
                self.val_dataloader.dataset.update_end_transforms()
        
        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            wandb.run.finish()

        return self.model
    
    @torch.no_grad()
    def evaluate(self, epoch: int, reduction: str = 'mean', wandb_flag: bool = False, returns: str = 'all',
                 custom_val_dataloader = None) -> float:
        
        self.model.eval()

        header = '[VALIDATION] - Epoch: [{}]'.format(epoch)

        batches_losses = []
        batches_losses_all = []
        if custom_val_dataloader is not None:
            dl = custom_val_dataloader
        else:
            dl = self.val_dataloader
        for i, (images, targets) in enumerate(self.val_logger.log_every(dl, self.print_freq, header)):

            images, targets = self.prepare_data(images, targets)

            output, loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            if returns != 'all':
                losses = loss_dict[returns]

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.val_logger.update(loss=losses_reduced, **loss_dict_reduced)

            batches_losses.append(losses)
            batches_losses_all.append(loss_dict_reduced)

            # if wandb_flag and self.vizual_fn is not None:
            #     if (i % self.print_freq == 0 or i == len(self.val_dataloader) - 1):
            #         fig = self._vizual(image = images, target = targets, output = output)
            #         wandb.log({'validation_vizuals': fig})
        
        batches_losses = torch.stack(batches_losses)


        if reduction == 'mean':
            out = torch.mean(batches_losses).item()

            # Get all keys from first dict
            avg_losses = {
                key: torch.stack([loss_dict[key] for loss_dict in batches_losses_all]).mean().item()
                for key in batches_losses_all[0].keys()
            }
            avg_losses["total_loss"] = out

            logger.info(f'{header} mean loss: {out:.4f} and other losses: {avg_losses}')

            return avg_losses
        
        elif reduction == 'sum':
            out = torch.sum(batches_losses).item()
            logger.info(f'{header} sum loss: {out:.4f}')

            return out

    def _train(
        self, 
        epoch: int, 
        warmup_iters: Optional[int] = None, 
        wandb_flag: bool = False
        ) -> torch.Tensor:
        ''' Training method '''

        self.model.train()

        # if hasattr(self.model, 'backbone'):
        #     self.model.backbone.eval()

        self.train_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = '[TRAINING] - Epoch: [{}]'.format(epoch)

        if warmup_iters is not None and epoch == 1:
            self.start_lr_scheduler = self._warmup_lr_scheduler(
                min(warmup_iters, len(self.train_dataloader)-1), 
                1. / warmup_iters
                )

        batches_losses = []

        for images, targets in self.train_logger.log_every(self.train_dataloader, self.print_freq, header):

            images, targets = self.prepare_data(images, targets)

            self.optimizer.zero_grad()

            loss_dict = self.model(images, targets)

            if wandb_flag:
                wandb.log(loss_dict)

            self.losses = sum(loss for loss in loss_dict.values())
            batches_losses.append(self.losses.detach())

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                logger.info(loss_dict_reduced)
                sys.exit(1)

            self.losses.backward()

            # Clip gradients to prevent explosions in the Transformer head
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            if self.adaloss is not None:
                self.adaloss.feed(self.losses)

            if warmup_iters is not None and epoch == 1:
                self.start_lr_scheduler.step()

            self.train_logger.update(loss=losses_reduced, **loss_dict_reduced)
            self.train_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        
        batches_losses = torch.stack(batches_losses)

        out = torch.mean(batches_losses).item()
        logger.info(f'{header} mean loss: {out:.4f}')

        return out
    
    def _warmup_lr_scheduler(self, warmup_iters: int, warmup_factor: float):
        ''' Method to make a warmup lr scheduler '''

        def warmup_func(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_func)
    
    def _lr_scheduler(self):
        ''' Method to make the lr scheduler '''

        if self.auto_lr is True:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        elif isinstance(self.auto_lr, dict):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.auto_lr)

        elif self.lr_milestones is not None:
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_milestones)
        
        else:
            return None
    
    def _prepare_evaluator(self, filename: str, epoch: int) -> None:
        ''' Evaluate the epoch model '''

        if self.evaluator is not None:
            self.evaluator.model = self.model
            self.evaluator.logs_filename = filename
            self.evaluator.header = '[{}] - Epoch: [{}]'.format(filename.upper(),epoch)
            self.evaluator.current_epoch = epoch
    
    def _is_best(self, val_output: float, mode: str = 'min') -> bool:
        ''' Method to determine the best model for saving checkpoint '''
        
        if mode == 'min':
            if val_output < self.best_val:
                self.best_val = val_output
                return True
            else:
                return False
        
        elif mode =='max':
            if val_output > self.best_val:
                self.best_val = val_output
                return True
            else:
                return False

    def _is_best_loss(self, val_output: float) -> bool:
        ''' Method to determine the best model for saving checkpoint '''

        if val_output < self.best_val_loss:
            self.best_val_loss = val_output
            return True
        else:
            return False

    
    def _save_checkpoint(self, epoch: int, mode: str) -> Path:
        ''' Method to save checkpoints '''

        check_dir = self.work_dir

        if mode == 'all':
            outpath = os.path.join(check_dir,f'epoch_{epoch}.pth')
        elif mode == 'best':
            outpath = os.path.join(check_dir,'best_model.pth')
        elif mode == 'best_loss':
            outpath = os.path.join(check_dir,'best_loss_model.pth')
        elif mode == 'latest':
            outpath = os.path.join(check_dir,'latest_model.pth')
        else:
            raise ValueError("wrong mode, should be 'all', 'best', 'best_loss','latest'")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses ,
            'best_val': self.best_val,

            }, outpath)

        return outpath
    
    def _vizual(self, image: Any, target: Any, output: Any):
        fig = self.vizual_fn(image=image, target=target, output=output)
        return fig

@TRAINERS.register()
class FasterRCNNTrainer(Trainer):
    ''' Class for training a Faster-RCNN model '''

    def prepare_data(self, images, targets) -> tuple:

        images = list(image.to(self.device) for image in images)
            
        targets = [{k: v.to(self.device) for k, v in t.items() if torch.is_tensor(v)} 
                            for t in targets]

        return images, targets


import torch
import math
import sys
from typing import Optional, List, Tuple
from loguru import logger


from ..utils.torchvision_utils import SmoothedValue, reduce_dict


# ... and other necessary imports from the base Trainer ...

# Note: The base Trainer class must be available for this subclass to work.

@TRAINERS.register()
class P2PNetTrainer(Trainer):
    '''
    Specialized Trainer for HerdNetP2P models.
    Skips the LossWrapper and correctly handles list-of-dictionary targets.
    '''

    def prepare_data(self, images, targets) -> tuple:
        """
        Overrides base method to handle the List-of-Dictionary targets specific to P2PNet,
        moving only the necessary Tensor components to the device.
        """
        images = images.to(self.device)

        if isinstance(targets, (list, tuple)):
            new_targets = []
            for t in targets:
                t_on_device = {}
                # Only move Tensors (points, labels) to the GPU
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t_on_device[k] = v.to(self.device)
                    else:
                        t_on_device[k] = v
                new_targets.append(t_on_device)
            targets = new_targets
        else:
            # Fallback for single tensor targets
            targets = targets.to(self.device)

        return images, targets

    def _train(
            self,
            epoch: int,
            warmup_iters: Optional[int] = None,
            wandb_flag: bool = False
    ) -> torch.Tensor:
        ''' Training method, calling HerdNetP2P directly for internal loss calculation. '''

        # Note: self.model is the raw HerdNetP2P, which has the loss criterion internally.
        self.model.train()

        # You must keep this check if DINO backbone evaluation is required
        if hasattr(self.model, 'backbone') and self.model.backbone.training:
            self.model.backbone.eval()  # Keep the batch norm/dropout frozen in DINO

        self.train_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = '[TRAINING] - Epoch: [{}]'.format(epoch)

        if warmup_iters is not None and epoch == 1:
            self.start_lr_scheduler = self._warmup_lr_scheduler(
                min(warmup_iters, len(self.train_dataloader) - 1),
                1. / warmup_iters
            )

        batches_losses = []

        for images, targets in self.train_logger.log_every(self.train_dataloader, self.print_freq, header):

            images, targets = self.prepare_data(images, targets)

            self.optimizer.zero_grad()

            # --- P2PNET FORWARD PASS ---
            # Call the model directly. HerdNetP2P calculates the Hungarian loss internally
            # and returns the scalar loss dict: {'loss_p2p': value}
            loss_dict = self.model(images, targets)

            if wandb_flag:
                wandb.log(loss_dict)

            # This sum is safe because loss_dict only contains scalar losses now
            self.losses = loss_dict["loss_p2p"]
            batches_losses.append(self.losses.detach())


            loss_value = self.losses.detach()

            if not math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                logger.info(self.losses.detach())
                sys.exit(1)

            self.losses.backward()

            # Clip gradients to prevent explosions in the Transformer head
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            if self.adaloss is not None:
                self.adaloss.feed(self.losses)

            if warmup_iters is not None and epoch == 1:
                self.start_lr_scheduler.step()

            self.train_logger.update(loss=self.losses.detach())
            self.train_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        batches_losses = torch.stack(batches_losses)

        out = torch.mean(batches_losses).item()
        logger.info(f'{header} mean loss: {out:.4f}')

        return out