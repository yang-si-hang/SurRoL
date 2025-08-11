import os
import time
import numpy as np

import pybullet as p 
from surrol.tasks.psm_env_full_dof import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)# load and define the MTM
import dvrk
import numpy as np
import rospy
import time
import math
# move in cartesian space
import PyKDL

from dvrk import mtm
class NeedlePickFullDof_haptic(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.695, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1, random_seed=1024):
        super(NeedlePickFullDof_haptic, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        self.random_seed = random_seed


    def _env_setup(self):
        super(NeedlePickFullDof_haptic, self)._env_setup()
        np.random.seed(self.random_seed)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        print(f"inverse kinemetics return number: {len(joint_positions)}")
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
        self.m = mtm('MTMR')

        # turn gravity compensation on/off
        self.m.use_gravity_compensation(True)
        self.m.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


        psm_pose = self.psm1.get_current_position()
        print(f"PSM pose RCM: {psm_pose}")
        print("PSM current orn", get_euler_from_matrix(psm_pose[:3,:3]))
        psm_pose_ori = psm_pose.copy()

        # psm_measured_cp = np.matmul(np.linalg.inv(ecm_pose), psm_pose_ori)#over ecm's rcm
        psm_measured_cp=psm_pose_ori #not over ecm
        # psm_measured_cp= np.matmul(psm1_transform,psm_measured_cp) #over ecm,then transform
        # psm_measured_cp = np.matmul(psm_pose_ori,psm1_transform)
        print(f"PSM  pose: {psm_measured_cp}")
        print(f"mtm orientation{self.m.setpoint_cp().M}")
        goal = PyKDL.Frame()
        goal.p = self.m.setpoint_cp().p
        # # goal.p[0] += 0.05
        goal.M= self.m.setpoint_cp().M

        # psm_measured_cp = np.matmul(mapping_mat,psm_measured_cp)
        for i in range(3):
            print(f"previous goal:{goal.M}")
            for j in range(3):
                goal.M[i,j]=psm_measured_cp[i][j]
                # if j==1:
                #     goal.M[i,j]*=-1
                # goal.M[i,j]=psm_pose[i][j]
            print(f"modified goal:{goal.M}")
        print(goal.M.GetEulerZYX())
        # print(rotationMatrixToEulerAngles(psm_measured_cp[:3,:3]))
        self.m.move_cp(goal).wait() #align
        print("orn after align: ",self.m.measured_cp().M.GetEulerZYX())
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        # self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        # self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        # self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        # self._waypoints[3] = np.array([self.goal[0], self.goal[1],
        #                                self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(7)
        action[6] = -0.5
        # for i, waypoint in enumerate(self._waypoints):
        #     if waypoint is None:
        #         continue
        #     delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
        #     delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
        #     if np.abs(delta_pos).max() > 1:
        #         delta_pos /= np.abs(delta_pos).max()
        #     scale_factor = 0.4
        #     delta_pos *= scale_factor
        #     action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
        #     if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
        #         self._waypoints[i] = None
        #     break

        return action
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])
    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)


if __name__ == "__main__":
    env = NeedlePickFullDof_haptic(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
