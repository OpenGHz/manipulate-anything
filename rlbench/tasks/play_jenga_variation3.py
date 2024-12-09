from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import Condition, DetectedCondition, NothingGrasped
from rlbench.backend.task import Task


class PlayJengaVariation3(Task):
    def init_task(self) -> None:
        target = Shape("target_cuboid")
        original_detector = ProximitySensor("original_detector")
        target_detector = ProximitySensor("target_detector")
        bricks = [Shape("Cuboid%d" % i) for i in range(13)]
        conds: list[Condition] = [
            DetectedCondition(b, original_detector) for b in bricks
        ]
        conds.extend(
            [
                DetectedCondition(target, target_detector),
                NothingGrasped(self.robot.gripper),  # type: ignore
            ]
        )
        self.register_success_conditions(conds)
        self.register_graspable_objects([target])

    def init_episode(self, index: int) -> List[str]:
        return ["Take the jenga block and place it on top of the jenga tower"]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]
