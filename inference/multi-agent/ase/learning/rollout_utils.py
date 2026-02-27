import hydra
import torch
import sys
from copy import deepcopy
from omegaconf import OmegaConf
sys.path.append('./')
from ase.learning import torch_utils

def load_checkpoint(payload):
    cfg = payload['cfg']

    cfg.dataset.motion_path = "./data/template.pkl"#cfg.dataset.motion_path.replace('/nas/nas_38/AI-being', '/mnt/nas_38')
    # cfg.dataset.text_path = '/mnt/nas_38/libin/Interhuman/annots'#cfg.dataset.text_path.replace('/nas/nas_38/AI-being', '/mnt/nas_38')
    if 'multi_gpu' in cfg._target_ :
        cfg._target_ = 'training.interagent.t2m_workspace.DiffusionPolicyWorkspace'
        cfg.dataset._target_ = 'training.interagent.dataset.dataset_two.DiffusionPolicyDataset'
        cfg.policy._target_ = 'training.interagent.policy.DiffusionPolicy'
        cfg.policy.model._target_ = 'training.interagent.modules.TransformerForDiffusion'

        state_dicts = payload['state_dicts']

        model_params = deepcopy(state_dicts['model'])
        ema_model_params = deepcopy(state_dicts['ema_model'])

        payload['state_dicts']['model'] = {k.replace("module.", ""): v for k, v in model_params.items() if "module" in k}
        payload['state_dicts']['ema_model'] = {k.replace("module.", ""): v for k, v in ema_model_params.items() if "module" in k}


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    training_dict['device'] = device
    cfg.update({'training':training_dict})

    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    return policy

def compute_interaction_graph_obs(rigid_body_pos,rigid_body_rot,intergraph_obs_buf):
    # intergraph_obs_buf = torch.zeros((8, 2, 15, 15, 9))
    # intergraph_body_ids = #[ 1,  2,  5,  8, 11, 14,  4,  7]
    # print('body_pos',rigid_body_pos.shape)
    _, _, J1, J2, _ = intergraph_obs_buf.shape

    body_pos = rigid_body_pos
    body_rot = rigid_body_rot

    nenvs = body_pos.shape[0]
    # [num_envs, 15, 1, 3] - agent 0's bodies as reference
    #agent0_pos = body_pos[:, 0].unsqueeze(2)
    agent0_pos_simply = body_pos[:, 0].unsqueeze(2)
    # [num_envs, 1, 15, 3] - agent 1's bodies as targets
    agent1_pos = body_pos[:, 1].unsqueeze(1)
    # print('agent0_pos_simply',agent0_pos_simply.shape)
    # print('agent1_pos',agent1_pos.shape)

    agent0_rot_simply = body_rot[:, 0].unsqueeze(2).expand(-1, J1, J2, -1)
    agent1_rot = body_rot[:, 1].unsqueeze(1).expand(-1, J1, J2, -1)
    # print('agent0_rot_simply',agent0_rot_simply.shape)
    # print('agent1_rot',agent1_rot.shape)
    rel_quat_agent0 = torch_utils.quat_diff(agent0_rot_simply.reshape(-1, 4), agent1_rot.reshape(-1, 4))
    rel_quat_agent0 = torch_utils.quat_to_tan_norm(rel_quat_agent0)
    rel_quat_agent0 = rel_quat_agent0.reshape(nenvs, J1, J2, -1)
    # print('rel_quat_agent0',rel_quat_agent0.shape)


    intergraph_obs_buf[:, 0,:,:,:3] = agent1_pos - agent0_pos_simply
    intergraph_obs_buf[:, 0,:,:,3:] = rel_quat_agent0

    agent1_pos_simply = body_pos[:, 1].unsqueeze(2)
    agent0_pos = body_pos[:, 0].unsqueeze(1)
    agent1_rot_simply = body_rot[:, 1].unsqueeze(2).expand(-1, J1, J2, -1)
    agent0_rot = body_rot[:, 0].unsqueeze(1).expand(-1, J1, J2, -1)
    rel_quat_agent1 = torch_utils.quat_diff(agent1_rot_simply.reshape(-1, 4), agent0_rot.reshape(-1, 4))
    rel_quat_agent1 = torch_utils.quat_to_tan_norm(rel_quat_agent1)
    rel_quat_agent1 = rel_quat_agent1.reshape(nenvs, J1, J2, -1)

    intergraph_obs_buf[:, 1,:,:,:3] = agent0_pos - agent1_pos_simply
    intergraph_obs_buf[:, 1,:,:,3:] = rel_quat_agent1
    return intergraph_obs_buf




def compute_interaction_graph_obs_pos(rigid_body_pos,rigid_body_rot,intergraph_obs_buf):
    # intergraph_obs_buf = torch.zeros((8, 2, 15, 15, 9))
    # intergraph_body_ids = #[ 1,  2,  5,  8, 11, 14,  4,  7]
    # print('body_pos',rigid_body_pos.shape)
    _, _, J1, J2, _ = intergraph_obs_buf.shape

    body_pos = rigid_body_pos
    body_rot = rigid_body_rot

    nenvs = body_pos.shape[0]
    # [num_envs, 15, 1, 3] - agent 0's bodies as reference
    #agent0_pos = body_pos[:, 0].unsqueeze(2)
    agent0_pos_simply = body_pos[:, 0].unsqueeze(2)
    # [num_envs, 1, 15, 3] - agent 1's bodies as targets
    agent1_pos = body_pos[:, 1].unsqueeze(1)
    # print('agent0_pos_simply',agent0_pos_simply.shape)
    # print('agent1_pos',agent1_pos.shape)

    # agent0_rot_simply = body_rot[:, 0].unsqueeze(2).expand(-1, J1, J2, -1)
    # agent1_rot = body_rot[:, 1].unsqueeze(1).expand(-1, J1, J2, -1)
    # print('agent0_rot_simply',agent0_rot_simply.shape)
    # print('agent1_rot',agent1_rot.shape)
    # rel_quat_agent0 = torch_utils.quat_diff(agent0_rot_simply.reshape(-1, 4), agent1_rot.reshape(-1, 4))
    # rel_quat_agent0 = torch_utils.quat_to_tan_norm(rel_quat_agent0)
    # rel_quat_agent0 = rel_quat_agent0.reshape(nenvs, J1, J2, -1)
    # # print('rel_quat_agent0',rel_quat_agent0.shape)


    intergraph_obs_buf[:, 0,:,:,:3] = agent1_pos - agent0_pos_simply
    # intergraph_obs_buf[:, 0,:,:,3:] = rel_quat_agent0

    agent1_pos_simply = body_pos[:, 1].unsqueeze(2)
    agent0_pos = body_pos[:, 0].unsqueeze(1)
    # agent1_rot_simply = body_rot[:, 1].unsqueeze(2).expand(-1, J1, J2, -1)
    # agent0_rot = body_rot[:, 0].unsqueeze(1).expand(-1, J1, J2, -1)
    # rel_quat_agent1 = torch_utils.quat_diff(agent1_rot_simply.reshape(-1, 4), agent0_rot.reshape(-1, 4))
    # rel_quat_agent1 = torch_utils.quat_to_tan_norm(rel_quat_agent1)
    # rel_quat_agent1 = rel_quat_agent1.reshape(nenvs, J1, J2, -1)

    intergraph_obs_buf[:, 1,:,:,:] = agent0_pos - agent1_pos_simply
    # intergraph_obs_buf[:, 1,:,:,3:] = rel_quat_agent1
    return intergraph_obs_buf


def compute_interaction_graph_obs_batch_pos_local(local_body_pos1, local_body_pos2):
    # rigid_body_pos: (B, F, 2, 15, 3)
    # rigid_body_rot: (B, F, 2, 15, 4)

    B, F, A, J, _ = local_body_pos1.shape  # A = 2 agents, J = 15 joints

    rigid_body_pos1 = local_body_pos1.reshape(B * F, A, J, 3)
    rigid_body_pos2 = local_body_pos2.reshape(B * F, A, J, 3)

    intergraph_obs_buf = torch.zeros((B * F, 2, J, J, 3), device=rigid_body_pos1.device)

    agent0_pos = rigid_body_pos1[:, 0].unsqueeze(2)  # (B*F, 15, 1, 3)
    agent1_pos = rigid_body_pos1[:, 1].unsqueeze(1)  # (B*F, 1, 15, 3)

    intergraph_obs_buf[:, 0, :, :, :] = agent1_pos - agent0_pos

    agent1_pos = rigid_body_pos2[:, 0].unsqueeze(2)
    agent0_pos = rigid_body_pos2[:, 1].unsqueeze(1)

    intergraph_obs_buf[:, 1, :, :, :] = agent0_pos - agent1_pos

    return intergraph_obs_buf.reshape(B, F, 2, J, J, 3)



def compute_interaction_graph_obs_batch_pos_rot_local(local_pos_1, local_pos_2, local_rot_1, local_rot_2):
    # rigid_body_pos: (B, F, 2, 15, 3)
    # rigid_body_rot: (B, F, 2, 15, 4)

    B, F, A, J, _ = local_pos_1.shape  # A = 2 agents, J = 15 joints

    rigid_body_pos1 = local_pos_1.reshape(B * F, A, J, 3)
    rigid_body_pos2 = local_pos_2.reshape(B * F, A, J, 3)

    rigid_body_rot1 = local_rot_1.reshape(B * F, A, J, 4)
    rigid_body_rot2 = local_rot_2.reshape(B * F, A, J, 4)

    intergraph_obs_buf = torch.zeros((B * F, 2, J, J, 9), device=rigid_body_pos1.device)

    agent0_pos = rigid_body_pos1[:, 0].unsqueeze(2)  # (B*F, 15, 1, 3)
    agent1_pos = rigid_body_pos1[:, 1].unsqueeze(1)  # (B*F, 1, 15, 3)
    agent0_rot = rigid_body_rot1[:, 0].unsqueeze(2).expand(-1, J, J, -1)
    agent1_rot = rigid_body_rot1[:, 1].unsqueeze(1).expand(-1, J, J, -1)
    rel_quat_01 = torch_utils.quat_diff(agent0_rot.reshape(-1, 4), agent1_rot.reshape(-1, 4))
    rel_quat_01 = torch_utils.quat_to_tan_norm(rel_quat_01).reshape(B * F, J, J, -1)

    intergraph_obs_buf[:, 0, :, :, :3] = agent1_pos - agent0_pos
    intergraph_obs_buf[:, 0, :, :, 3:] = rel_quat_01

    agent1_pos = rigid_body_pos2[:, 0].unsqueeze(2)
    agent0_pos = rigid_body_pos2[:, 1].unsqueeze(1)
    agent1_rot = rigid_body_rot2[:, 0].unsqueeze(2).expand(-1, J, J, -1)
    agent0_rot = rigid_body_rot2[:, 1].unsqueeze(1).expand(-1, J, J, -1)
    rel_quat_10 = torch_utils.quat_diff(agent1_rot.reshape(-1, 4), agent0_rot.reshape(-1, 4))
    rel_quat_10 = torch_utils.quat_to_tan_norm(rel_quat_10).reshape(B * F, J, J, -1)

    intergraph_obs_buf[:, 1, :, :, :3] = agent0_pos - agent1_pos
    intergraph_obs_buf[:, 1, :, :, 3:] = rel_quat_10

    return intergraph_obs_buf.reshape(B, F, 2, J, J, 9)

def compute_interaction_graph_obs_batch_pos_local_latent_ig(local_body_pos1, local_body_pos2):
    # rigid_body_pos: (B, F, 2, 15, 3)
    # rigid_body_rot: (B, F, 2, 15, 4)

    B, F, A, J, _ = local_body_pos1.shape  # A = 2 agents, J = 15 joints

    rigid_body_pos1 = local_body_pos1.reshape(B * F, A, J, 3)
    rigid_body_pos2 = local_body_pos2.reshape(B * F, A, J, 3)

    intergraph_obs_buf = torch.zeros((B * F, 2, J, J, 3), device=rigid_body_pos1.device)

    agent0_pos = rigid_body_pos1[:, 0].unsqueeze(2)  # (B*F, 15, 1, 3)
    agent1_pos = rigid_body_pos1[:, 1].unsqueeze(1)  # (B*F, 1, 15, 3)

    intergraph_obs_buf[:, 0, :, :, :] = agent1_pos - agent0_pos

    agent1_pos = rigid_body_pos2[:, 0].unsqueeze(2)
    agent0_pos = rigid_body_pos2[:, 1].unsqueeze(1)

    intergraph_obs_buf[:, 1, :, :, :] = agent0_pos - agent1_pos

    # 返回形状: [B*F, 2*J, J*3] = [B*F, 30, 45]
    # 这样可以在后续重新整理为 [B, F, features] 格式
    return intergraph_obs_buf.reshape(B * F, 2 * J, J * 3)