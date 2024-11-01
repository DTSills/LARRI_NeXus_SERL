from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import cv2
import numpy as np
import pandas as pd
from gym import spaces
import jax.numpy as jnp

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.envs.offline_picofuncs import get_state, build_index, get_img

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PICO_HOME = np.asarray((55, 350))
_IMAGE_SET = "C:/Users/samue/PycharmProjects/BlurDetection/Prt2_Combined"
_LINEW_CHOICE = np.asarray([0.1, 0.25, 0.5, 0.75, 1])
_CARTESIAN_BOUNDS = np.asarray([[50, 200], [61, 500]])


class PrinterGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([5, 100]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        # Caching Initial Data
        self.stroke, self.close = _PICO_HOME

        self.cycle = 0
        self.step_cnt = 0

        #Rendering and Caching commands from use with Mujoco Renderer

        # self.render_mode = render_mode
        # self.camera_id = (0, 1)
        # self.image_obs = image_obs
        #
        # # Caching.
        # self._panda_dof_ids = np.asarray(
        #     [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        # )
        # self._panda_ctrl_ids = np.asarray(
        #     [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        # )
        # self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        # self._pinch_site_id = self._model.site("pinch").id
        # self._block_z = self._model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "Satellites": spaces.Box(
                            0, np.inf, shape=(1,), dtype=np.int32
                        ),
                        "Width": spaces.Box(
                            0, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "Area_Diff": spaces.Box(
                                 0, np.inf, shape=(1,), dtype=np.float32
                        )
                    }
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1, -1]),
            high=np.asarray([1, 1]),
            dtype=np.float32,
        )

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        # from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        #
        # self._viewer = MujocoRenderer(
        #     self.model,
        #     self.data,
        # )
        # self._viewer.render(self.render_mode)

        # Generate state space index from file of images
        self.index = build_index(_IMAGE_SET)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # Reset to Picopulse Home
        self.stroke, self.close = _PICO_HOME

        self.cycle += 1
        self.step_cnt = 0

        # Sample a new line_w.
        line_w = np.random.choice(_LINEW_CHOICE)
        self.target = (1, line_w, 0)

        # Compute Starting State
        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        stroke, close_t = action

        # Set the stroke.
        stk = self.stroke
        dstk = stroke * self._action_scale[0]
        nstk = np.clip(stk + dstk, *_CARTESIAN_BOUNDS[:, 0])
        self.stroke = round(nstk)

        # Set the Close_t
        cls = self.close
        dcls = close_t * self._action_scale[1]
        ncls = np.clip(cls + dcls, *_CARTESIAN_BOUNDS[:, 1])
        self.close = round(ncls)

        # for _ in range(self._n_substeps):
        #     tau = opspace(
        #         model=self._model,
        #         data=self._data,
        #         site_id=self._pinch_site_id,
        #         dof_ids=self._panda_dof_ids,
        #         pos=self._data.mocap_pos[0],
        #         ori=self._data.mocap_quat[0],
        #         joint=_PANDA_HOME,
        #         gravity_comp=True,
        #     )
        #     self._data.ctrl[self._panda_ctrl_ids] = tau
        #     mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()
        self.step_cnt += 1

        return obs, rew, terminated, False, {}

    def render(self):
        image = get_img(self.stroke,self.close,self.index)
        ln1 = ("Target Line_W: " + str(self.target[1]))
        ln2 = ("Cycle: " + str(self.cycle) + " Step Count: " + str(self.step_cnt))
        ln3 = ("Rew: " + str(self._compute_reward()))
        image = cv2.putText(image, ln1, (0,25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2, cv2.LINE_AA)
        image = cv2.putText(image, ln2, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        image = cv2.putText(image, ln3, (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Current Image', image)
        cv2.waitKey(250)

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        satellites, dep_w, difference = get_state(self.stroke, self.close, self.index)

        obs["state"]["Satellites"] = float(satellites)

        obs["state"]["Width"] = dep_w

        obs["state"]["Area_Diff"] = difference

        # joint_pos = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_pos"] = joint_pos.astype(np.float32)

        # joint_vel = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_vel"] = joint_vel.astype(np.float32)

        # joint_torque = np.stack(
        # [self._data.sensor(f"panda/joint{i}_torque").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_torque"] = symlog(joint_torque.astype(np.float32))

        # wrist_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
        # obs["panda/wrist_force"] = symlog(wrist_force.astype(np.float32))

        #if self.image_obs:
        #    obs["images"] = {}
        #    obs["images"]["front"], obs["images"]["wrist"] = self.render()
        #else:
        #    block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        #    obs["state"]["block_pos"] = block_pos

        #if self.render_mode == "human":
        #    self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        ideal_pos = np.asarray(self.target)
        current_pos = np.asarray(get_state(self.stroke, self.close, self.index))
        dist = np.linalg.norm(ideal_pos - current_pos)
        r_close = np.exp(-20 * dist)
        rew = r_close
        return rew

    def close(self):
        # Not sure what needs to happen here but the default is bad...
        return


if __name__ == "__main__":
    env = PrinterGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 2))
        env.render()
    actor = [[env.observation_space.sample()], env.action_space.sample()]
    print(jnp.shape(actor)[-1])
    env.close()
