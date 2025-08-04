
from logging import getLogger

import torch
import numpy as np
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from LEHD_TSP_p2.TSP.TSPModel import TSPModel as Model
from LEHD_TSP_p2.TSP.test_TSPlib import main_test
from LEHD_TSP_p2.TSP.TSPEnv import TSPEnv as Env
from LEHD_TSP_p2.utils.utils import *


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):


        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        random_seed = 123
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # Main Components
        self.model = Model(**self.model_params)

        self.env = Env(**self.env_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/p1-checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if trainer_params['use_optimizer_state']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = 0
            self.logger.info('Saved Model Loaded !!')
        if trainer_params['gen_type']:
            self.gen_type = trainer_params['gen_type']
        if trainer_params['use_early_stopping']:
            self.tolerance = trainer_params['tolerance']

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'] )

        save_gap = []
        no_improvement_count = 0
        early_stopping_flag = False
        best_gap = None
        best_epoch = 0
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            if early_stopping_flag:
                self.logger.info(f"Early stopping triggered at epoch {epoch-1}. Best gap: {best_gap:.6f} at epoch {best_epoch}")
                break
            self.logger.info('=================================================================')
            # self.env.shuffle_data()
            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch)

            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']



            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),

                }
                torch.save(checkpoint_dict, '{}/p2-checkpoint-{}.pt'.format(self.result_folder, epoch))

                score_optimal, score_student ,gap = main_test(epoch,self.result_folder,use_RRC=False,gen_type=self.gen_type,
                                                              cuda_device_num=self.trainer_params['cuda_device_num'])

                save_gap.append([score_optimal, score_student,gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    no_improvement_count = 0
                    early_stopping_flag = False
                    best_epoch = epoch
                    self.logger.info(f"Epoch {epoch}: gap improved to {gap:.6f}")
                else:
                    no_improvement_count += 1
                    self.logger.info(f"Epoch {epoch}: gap {gap:.6f} (no improvement, count={no_improvement_count})")
                    if no_improvement_count >= self.tolerance:
                        early_stopping_flag = True
                        self.logger.info(f"Early stopping will trigger after this epoch! (no improvement in {self.tolerance} intervals)")

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")


    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        max_batch_size = self.trainer_params['max_batch_size']

        # Generate and shuffle the index
        problem_indices = np.arange(train_num_episode)
        np.random.shuffle(problem_indices)

        for i, ep in enumerate(problem_indices):
            avg_score, score_student_mean, avg_loss, problem_batch_size = self._train_one_batch(episode=ep, max_batch_size=max_batch_size, epoch=epoch)
            score_AM.update(avg_score, problem_batch_size)
            score_student_AM.update(score_student_mean, problem_batch_size)
            loss_AM.update(avg_loss, problem_batch_size)
            
            self.logger.info(
                f'Epoch {epoch:3d}: Train {i+1:3d}/{train_num_episode} ({100. * (i+1) / train_num_episode:1.1f}%)  '
                f'Score: {score_AM.avg:.4f}, Score_student: {score_student_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}'
            )
            
        # self.logger.info(
        #     f'Epoch {epoch:3d}: Train (100%)  Score: {score_AM.avg:.4f}, Score_student: {score_student_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}'
        # )
        return score_AM.avg, score_student_AM.avg, loss_AM.avg


    def _train_one_batch(self, episode,max_batch_size,epoch):

        ###############################################
        self.model.train()
        problem_batch_size, problem_size = self.env.load_problems(episode,max_batch_size)
        reset_state, _, _ = self.env.reset(self.env_params['mode'])
        
        self.logger.info(f'Training problem with problem size {problem_size} with batch size{problem_batch_size}...')
        
        prob_list = torch.ones(size=(problem_batch_size, 0))
        state, reward,reward_student, done = self.env.pre_step()

        current_step=0

        while not done:
            if current_step == 0:
                selected_teacher = self.env.solution[:, -1] # destination node
                selected_student = self.env.solution[:, -1]
                prob = torch.ones(self.env.solution.shape[0], 1)
            elif current_step == 1:
                selected_teacher = self.env.solution[:, 0] # starting node
                selected_student = self.env.solution[:, 0]
                prob = torch.ones(self.env.solution.shape[0], 1)

            else:
                selected_teacher, prob, probs, selected_student = self.model(state, self.env.selected_node_list, self.env.solution, current_step)  # Update the selected points and probabilities
                loss_mean = -prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step+=1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

            prob_list = torch.cat((prob_list, prob), dim=1)

        loss_mean = -prob_list.log().mean()

        return 0,0, loss_mean.item(), problem_batch_size
