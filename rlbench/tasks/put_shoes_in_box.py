from typing import List, Tuple
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy

from rlbench.backend.conditions import (
    DetectedSeveralCondition, FalseCondition, NothingGrasped
)   
GROCERY_NAMES = [
    'shoe2'
    # 'strawberry jello',
    # 'soup',
    # 'spam',
    # 'mustard',
    # 'sugar',
]
SEG_IDS = {
    'shoe2': 82
}
PLACE_REGION_ID = 93

class PutShoesInBox(Task):

    def init_task(self):
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        self.register_graspable_objects([self.groceries])
        self.seg_ids = [SEG_IDS[name] for name in GROCERY_NAMES]
        self.place_region_id = PLACE_REGION_ID
        self.register_success_conditions([FalseCondition()])
        
        # success_sensor = ProximitySensor('success_in_box')
        # self.register_success_conditions([
        #     DetectedCondition(shoe1, success_sensor),
        #     DetectedCondition(shoe2, success_sensor),
        #     NothingGrasped(self.robot.gripper)])

    def init_episode(self, index: int) -> List[str]:
        
        return ['put the shoes in the box',
                'open the box and place the shoes inside',
                'open the box lid and put the shoes inside']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]
