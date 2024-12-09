from typing import List

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class SortItemsVariation2(Task):
    def init_task(self) -> None:
        self._object = Shape("tennis_ball")
        self._target_detector = ProximitySensor("green_patch_detector")

        self.register_graspable_objects([self._object])

        self.register_success_conditions(
            [
                DetectedCondition(self._object, self._target_detector),
            ]
        )

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return ["pick the tennis ball and place it into the green target"]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1
