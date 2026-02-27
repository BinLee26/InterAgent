import os
import copy
import random
import collections
from functools import partial

import torch
import numpy as np
import mujoco
from scipy.interpolate import interp1d
from gym import utils
from gym.utils.renderer import Renderer
from gym.envs.mujoco import Viewer, RenderContextOffscreen

if os.environ.get('DISPLAY', None) is None:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.5)),
    "elevation": 0.0,
    "azimuth": 180.0,
}
DEFAULT_SIZE = 480

# Config for initializing the Skeleton Mujoco environment
ENV_CONFIG = {
    'env_id': 'Skeleton',
    'xml_file': 'assets/S2_model.xml',
    'render_mode': 'rgb_array',
    'init_type': 'init_dist', 
    'init_path': 'motions/S07DN_init_dist',
    'frame_skip': 2,

    'max_ep_time': 6,
    'pert': {
        'active': True,

        # Time at which the perturbation starts
        'imp_time': 0, 

        # Percent of body weight to apply as a perturbation. The applied force is p_frc_frac * 59 Kg * 9.81 m/s^2.
        'p_frc_frac': 0.15,

        # The angle (degrees) FROM which the perturbation force is applied. 0 degrees is the forward direction
        # of the skeleton, so p_ang = 0 will apply the force from the front (toward the back) of the skeleton.
        # p_ang = 90 will apply a force from the left (toward the right) of the skeleton.
        'p_ang': 90,

        # Duration of the perturbation
        'p_dur': .3,
    },
    
    'rand_pert':{
        'active': False,
        'imp_time': [0, 2], 
        'p_frc_frac': [0.0745, 0.15],
        'p_dur': .3,
    },

    # Mujoco bodies to include in the obs
    'obs_mj_body_names': [
        'pelvis',
        'femur_r', 'tibia_r', 'calcn_r', 'toes_r',
        'femur_l', 'tibia_l', 'calcn_l', 'toes_l',
        'torso',
    ]
}


class MujocoEnv:
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        render_mode=None,
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.frame_skip = frame_skip
        self._initialize_simulation()
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}
        self.viewer = None

        # defined metadata here and removed asserts, annoying to redefine render_fps everytime...
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.render_mode = render_mode
        render_frame = partial(
            self._render,
            width=width,
            height=height,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.renderer = Renderer(self.render_mode, render_frame)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel=0):
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)  # nframes defined as frame_skip for some reason... force it to be 1.
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        assert mode in self.metadata["render_modes"]

        if mode in {
            "rgb_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def render(
        self,
        mode="human",
        width=None,
        height=None,
        camera_id=None,
        camera_name=None,
    ):
        if self.render_mode is not None:
            assert (
                width is None
                and height is None
                and camera_id is None
                and camera_name is None
            ), "Unexpected argument for render. Specify render arguments at environment initialization."
            return self.renderer.get_renders()
        else:
            width = width if width is not None else DEFAULT_SIZE
            height = height if height is not None else DEFAULT_SIZE
            return self._render(
                mode=mode,
                width=width,
                height=height,
                camera_id=camera_id,
                camera_name=camera_name,
            )

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = Viewer(self.model, self.data)
            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected mmodes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


class Skeleton(MujocoEnv):
    def __init__(self, cfg=ENV_CONFIG, **kwargs):
        self.cfg = cfg
        self.xml_file = self.cfg['xml_file']
        self.frame_skip = self.cfg['frame_skip']
        self.render_mode = self.cfg['render_mode']
        self.max_ep_time = self.cfg['max_ep_time']
        MujocoEnv.__init__(self, self.xml_file, self.frame_skip, render_mode=self.render_mode, **kwargs)

        # Path to file that contains the initial distribution of qpos and qvel
        self.init_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg['init_path']
        )
        assert os.path.exists(self.init_path), f"init_path {self.init_path} does not exist"
        self.init_type = self.cfg['init_type']
        assert self.init_type in ['init_dist']
        self.maybe_load_init_dist(self.init_path)

        # Perturbation parameters
        self.p_active = self.cfg['pert']['active']
        self.rand_pert_active = self.cfg['rand_pert']['active']
        assert not (self.p_active and self.rand_pert_active), "Only one perturbation type can be active at a time"
        
        self.obs_mj_body_names = self.cfg['obs_mj_body_names']

    @property
    def action_dim(self):
        return self.model.nu

    @property
    def mj_body_names(self):
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
    
    @property
    def obs_mj_body_idxs(self):
        return [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.obs_mj_body_names
        ]
        
    def maybe_load_init_dist(self, path):
        if self.init_type == 'init_dist':
            assert path is not None
            self.init_qpos_mean = np.load(path + '/init_qpos_mean.npy')
            self.init_qpos_std = np.load(path + '/init_qpos_std.npy')
            self.init_qvel_mean = np.load(path + '/init_qvel_mean.npy')
            self.init_qvel_std = np.load(path + '/init_qvel_std.npy')
            self.init_qvel_std[np.where(self.init_qvel_std == 0)] = .1

    def reset_model(self):
        # Handle perturbation parameters
        if self.p_active:
            self.p_frc_frac = self.cfg['pert']['p_frc_frac']
            self.p_frc = 9.81 * 59 * self.p_frc_frac
            self.p_ang = self.cfg['pert']['p_ang']
            self.p_dur = self.cfg['pert']['p_dur']
            self.imp_time = self.cfg['pert']['imp_time']
        elif self.rand_pert_active:
            self.rand_p_frc_frac_range = self.cfg['rand_pert']['p_frc_frac']
            self.p_frc_frac = np.random.uniform(*self.rand_p_frc_frac_range)
            p_frc = 9.81 * 59 * self.p_frc_frac
            self.p_frc = np.random.choice([self.p_frc, -self.p_frc])
            self.p_ang = None
            self.p_dur = self.cfg['rand_pert']['p_dur']
            self.rand_imp_time_range = self.cfg['rand_pert']['imp_time']
            self.imp_time = np.random.uniform(*self.rand_imp_time_range)

        # Handle initial state
        if self.init_type == 'init_dist':
            # This is for checking how diverse the policy can be given the identical starting state. add some noise when we test for robustness.
            qpos = np.random.normal(self.init_qpos_mean, self.init_qpos_std)
            qvel = np.random.normal(self.init_qvel_mean, self.init_qvel_std)
        self.set_state(qpos, qvel)

        # State machine info
        self.force_already_applied = False
        self.force_being_applied = False
        self.first_left_contact = False
        self.first_right_contact = False
        self.signal = 0
        
        self.left_foot_first_contact_pos = np.array([np.inf, np.inf]) 
        self.right_foot_first_contact_pos = np.array([np.inf, np.inf])
        self.left_foot_first_contact_root_pos = np.array([np.inf, np.inf])
        self.right_foot_first_contact_root_pos = np.array([np.inf, np.inf])

        self.data.qfrc_applied[:] = 0
        self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([.7, .5, .3, 1])

        return self.get_obs()

    def reset(self):
        self._reset_simulation()
        obs = self.reset_model()
        self.renderer.reset()
        self.renderer.render_step()
        return obs

    def get_done(self):
        done = self.data.body('torso').xpos[2] <= .6
        done = done or self.data.time > self.max_ep_time
        return done

    def get_obs(self):
        idxs = self.obs_mj_body_idxs
        xipos = self.data.xipos[idxs]
        ximat = self.data.ximat[idxs]
        cvel = self.data.cvel[idxs]

        # Signal for inference
        self.signal = 1 if self.force_being_applied else self.signal * 0.85
        self.signal = self.signal if self.signal >= 1e-3 else 0
        
        obs = np.hstack((xipos.flatten(), ximat.flatten(), cvel.flatten(), self.signal))
        return obs

    def update_foot_contact_positions(self):
        # returns the first 
        lhs = self.data.body('calcn_l').cfrc_ext[5]
        rhs = self.data.body('calcn_r').cfrc_ext[5]
        if self.data.time > .1 and self.force_already_applied:
            if lhs and not self.first_left_contact: 
                x_pos = self.data.body('calcn_l').xpos[0] # x, y (skel env defined in y up coordinate system)
                y_pos = -self.data.body('calcn_l').xpos[1] # x, y (skel env defined in y up coordinate system)
                left_foot = np.array([y_pos, x_pos])

                x_root = self.data.body('pelvis').xpos[0]
                y_root = -self.data.body('pelvis').xpos[1]
                root_pos = np.array([y_root, x_root])
                self.left_foot_first_contact_pos = left_foot.copy()
                self.left_foot_first_contact_root_pos = root_pos.copy()
                self.first_left_contact = True

            if rhs and self.first_left_contact and not self.first_right_contact: 
                x_pos = self.data.body('calcn_r').xpos[0] # x, y (skel env defined in y up coordinate system)
                y_pos = -self.data.body('calcn_r').xpos[1] # x, y (skel env defined in y up coordinate system)
                right_foot = np.array([y_pos, x_pos])

                x_root = self.data.body('pelvis').xpos[0]
                y_root = -self.data.body('pelvis').xpos[1]
                root_pos = np.array([y_root, x_root])
                self.right_foot_first_contact_pos = right_foot.copy()
                self.right_foot_first_contact_root_pos = root_pos.copy()
                self.first_right_contact = True

    def apply_force_sim(self, imp_time=None):
        assert self.imp_time is not None
        if self.imp_time <= self.data.time <= self.imp_time + 0.3:
            self.data.qfrc_applied[0] = np.cos(self.p_ang * np.pi / 180) * self.p_frc
            self.data.qfrc_applied[2] = np.sin(self.p_ang * np.pi / 180) * self.p_frc
            self.model.geom_rgba[[1, 2, 22], :] = np.array([1, 1, 1, 1])
            self.force_being_applied = True
        else:
            self.data.qfrc_applied[:] = 0
            self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([.7, .5, .3, 1])
            self.force_being_applied = False
            if self.data.time > self.imp_time + .3:
                self.force_already_applied = True

    def compute_torque(self, qtarget):
        qpos = self.data.qpos[6:-1].copy()
        qvel = self.data.qvel[6:-1].copy()
        kp, kd = 1, 0.05
        torque = kp * (qtarget - qpos) - kd * qvel
        return torque
    
    def step(self, action):
        assert isinstance(action, np.ndarray) and action.shape == (self.action_dim,)

        for _ in range(self.frame_skip):
            torque = self.compute_torque(action)
            self.data.qvel[-1] = -1.25 # Treadmill

            self._step_mujoco_simulation(ctrl=torque, n_frames=1)
            if self.p_active or self.rand_pert_active:
                self.apply_force_sim(imp_time=None)

        self.update_foot_contact_positions()
        self.renderer.render_step()
        obs = self.get_obs()
        reward = None
        done = self.get_done()
        info = {}
        if self.render_mode is not None:
            rgb = self.render(mode=self.render_mode)
            info['rgb'] = rgb[0].astype(np.uint8)

        return obs, reward, done, info
