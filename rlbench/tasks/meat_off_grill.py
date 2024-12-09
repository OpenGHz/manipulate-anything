from typing import List
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import (
    Condition,
    NothingGrasped,
    DetectedCondition,
)
from rlbench.backend.task import Task


class MeatOffGrill(Task):
    def init_task(self) -> None:
        self._steak = Shape("steak")
        self._success_sensor = ProximitySensor("success")
        self.register_graspable_objects([self._steak])
        self._w1 = Dummy("waypoint1")
        self._w1z = self._w1.get_position()[2]

    def init_episode(self, index: int) -> List[str]:
        conditions: List[Condition] = [NothingGrasped(self.robot.gripper)]
        x, y, _ = self._steak.get_position()
        self._w1.set_position([x, y, self._w1z])
        conditions.append(DetectedCondition(self._steak, self._success_sensor))
        self.register_success_conditions(conditions)
        return [
            "take the steak off the grill",
            "pick up the steak and place it next to the grill",
            "remove the steak from the grill and set it down to the side",
        ]

    def variation_count(self) -> int:
        return 1
