from typing import List

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class MultiTaskKitchen(Task):
    def init_task(self) -> None:
        self._cup = Shape("cup")
        self._plate1 = Shape("plate1")
        self._chicken = Shape("chicken")
        self.register_graspable_objects(
            [self._cup, self._plate1, self._chicken]
        )

        self.register_success_conditions(
            [
                DetectedCondition(
                    self._cup, ProximitySensor("success_sensor_cup")
                ),
                DetectedCondition(
                    self._plate1, ProximitySensor("success_sensor_plate")
                ),
                DetectedCondition(
                    self._chicken, ProximitySensor("success_sensor_plate")
                ),
            ]
        )

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return [""]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass
