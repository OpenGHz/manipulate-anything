from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedSeveralCondition, FalseCondition, NothingGrasped
)   
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
import random

GROCERY_NAMES = [
    'chocolate jello'
    # 'chocolate jello',
    # 'strawberry jello',
    # 'soup',
    # 'spam',
    # 'mustard',
    # 'sugar',
]
SEG_IDS = {
    'chocolate jello': 82,
    'strawberry jello': 85,
    'spam': 88,
    'sugar': 91,
    'crackers': 94,
    'mustartd': 97,
    'soup': 123
}
PLACE_REGION_ID = 119


class PickOneYcbPlace(Task):

    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        self.grasp_points = [Dummy('%s_grasp_point' % name.replace(' ', '_'))
                             for name in GROCERY_NAMES]
        self.goals = [Dummy('goal_%s' % name.replace(' ', '_'))
                      for name in GROCERY_NAMES]
        self.seg_ids = [SEG_IDS[name] for name in GROCERY_NAMES]
        self.place_region_id = PLACE_REGION_ID
        self.waypoint1 = Dummy('waypoint1')
        self.waypoint3 = Dummy('waypoint3')
        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape('groceries_boundary')])

        self.register_waypoint_ability_start(0, self._move_to_next_target)
        self.register_waypoint_ability_start(3, self._move_to_drop_zone)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        self.groceries_placed = 0
        self.boundary.clear()
        target_pose = np.array([ 1.502e-01 ,-2.79e-01, 8.55e-01, -1.3351e-05, 1.1809e-06, 9.971e-01, -7.6052e-02])
        self.groceries[0].set_pose(target_pose)

        # [self.boundary.sample(g, min_distance=0.15) for g in self.groceries]
        # print(self.groceries[0].get_pose())
        # print('*****')
        self.register_success_conditions([FalseCondition()])
        # self.register_success_conditions(
        #     [DetectedSeveralCondition(self.groceries[index:index+1], ProximitySensor('success'), 1),
        #      NothingGrasped(self.robot.gripper)])
        return ['put %s in the cupboard' % GROCERY_NAMES[index],
                'pick up %s and place it in the cupboard' % GROCERY_NAMES[index],
                'move %s to the shelves' % GROCERY_NAMES[index],
                'put %s on the table into the cupboard' % GROCERY_NAMES[index],
                'put away %s in the cupboard' % GROCERY_NAMES[index]]

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def is_static_workspace(self) -> bool:
        return True

    def _move_to_next_target(self, _):
        # input("Press Enter to extract to the first keypoint...")

        if self.groceries_placed >= self.groceries_to_place:
            raise RuntimeError('Should not be here.')
        # self.waypoint1.set_pose(
        #     self.grasp_points[self.groceries_placed].get_pose())
        print(self.grasp_points[self.groceries_placed].get_pose())
        # print(type(self.grasp_points[self.groceries_placed].get_pose()))
        # print(len(self.grasp_points[self.groceries_placed].get_pose()))
        old_pose=self.grasp_points[self.groceries_placed].get_pose()
        input_pose=[[0.17755932, -0.24042489, 0.94222283, 0.37978463523798606, -0.9240890901258848, -0.04246546150206974, 0.00444166233607271],
                    [-0.23313332, -0.049538735, 0.8527052, 0.37919693437657986, -0.880364390573617, 0.24076714628560836, 0.1523136514012233],
                    [0.16984509, -0.21358147, 0.93710357, 0.3573088188607966, -0.9196479290115576, -0.16216698171748076, -0.016731776546102126],
                    [0.17284057, -0.23586503, 0.93873537, 0.37098360210696124, -0.9252413825191157, -0.07789948228397546, 0.015205157663894655],
                    [0.1767463, -0.2364382, 0.938192, 0.3745560428113578, -0.9257536925307621, -0.05121080292419603, 0.00808526773645643],
                    [0.1728827, -0.20384602, 0.9190368, -0.35546189004043693, -0.927942975942783, -0.050714652829066925, -0.0999836212531163]
        ]
        new_set = np.array(input_pose[random.randint(0, 6)])
        old_pose[:] = new_set
        # print(pose)
        print(old_pose)
        self.waypoint1.set_pose(old_pose)

    def _move_to_drop_zone(self, _):
        self.waypoint3.set_pose(
            self.goals[self.groceries_placed].get_pose())

    def _repeat(self):
        self.groceries_placed += 1
        return self.groceries_placed < self.groceries_to_place