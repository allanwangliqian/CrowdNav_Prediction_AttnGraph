import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.sgan.models import TrajectoryGenerator
from sgan.sgan.utils import relative_to_abs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class SGANInference(object):

    def __init__(self, model_path, device, max_humans, num_envs):
        # To initialize it, a path to a pretrained model is needed
        # models are stored in sgan/models
        # for example: model_path = "models/sgan-models/eth_8_model.pt"
        #
        # model checkpoint names are like this
        # (dataset name)_(observation length)_model.pt
        #
        # dataset_name is the dataset that is the test set 
        # (the dataset that was not seen during the training of this model)
        #
        # obervation_length is the input length of the trajectory to this model

        path = model_path

        # number of samples to draw to get the final predicted trajectory
        self.num_samples = 20

        self.device = device
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        self.generator = self.get_generator(checkpoint)
        self.args = AttrDict(checkpoint['args'])

        seq_starts = np.array([i for i in range(0, max_humans * num_envs, max_humans)])
        seq_ends = seq_starts + max_humans
        seq_start_end = np.stack([seq_starts, seq_ends], axis=1)
        self.seq_start_end = torch.from_numpy(seq_start_end).type(torch.int)
        self.seq_start_end = self.seq_start_end.to(self.device)
        return

    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.to(self.device)
        generator.eval()
        return generator

    def evaluate(self, input_traj, input_binary_mask):
        # inputs:
        # depending on the observation length of your chosen model
        # the input obs_traj should be a numpy array of size nx8x2
        # where n is the number of people in the scene
        # obs_traj is simply the observed trajectories (sequence of coordinates)
        #
        # outputs:
        # outputs nx8x2 or nx12x2 predicted trajectories (sequence of coordinates)

        invalid_value = -999.
        # *** Process input data
        obs_traj = input_traj.permute(0,1,3,2) # (n_env, num_peds, 2, obs_seq_len)
        n_env, num_peds = obs_traj.shape[:2]
        loss_mask_obs = input_binary_mask[:,:,:,0] # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_obs = loss_mask_obs[:,:,:-1] * loss_mask_obs[:,:,-1:]
        loss_mask_rel_obs = torch.cat((loss_mask_obs[:,:,:1], loss_mask_rel_obs), dim=2) # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_pred = (torch.ones((n_env, num_peds, 8), device=self.device) * loss_mask_rel_obs[:,:,-1:])
        loss_mask_pred = loss_mask_rel_pred
        output_binary_mask = loss_mask_pred[:,:,:1].detach().to(self.device)

        obs_traj = input_traj.view(-1, 8, 2)
        num_entries = obs_traj.shape[0]
        traj_length = obs_traj.shape[1]
        obs_traj_rel = obs_traj[:, 1:traj_length] - obs_traj[:, 0:traj_length - 1]
        obs_traj_rel = torch.cat((torch.tensor([[[0,0]]] * num_entries, device=self.device), obs_traj_rel), axis=1)
        obs_traj = obs_traj.permute(1, 0, 2)
        obs_traj_rel = obs_traj_rel.permute(1, 0, 2)
        seq_start_end = self.seq_start_end

        with torch.no_grad():
            obs_traj = obs_traj.to(self.device)
            obs_traj_rel = obs_traj_rel.to(self.device)

            pred_traj_avg = torch.zeros((self.num_samples, obs_traj.size(1), obs_traj.size(0), 2), device=self.device)
            for i in range(self.num_samples):
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                pred_traj_avg[i] = pred_traj_fake.permute(1, 0, 2)

            pred_traj_avg = torch.mean(pred_traj_avg, dim=0)
            output_traj = torch.stack(torch.split(pred_traj_avg, num_peds, dim=0))

        return output_traj, output_binary_mask


