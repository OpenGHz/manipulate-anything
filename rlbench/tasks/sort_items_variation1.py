from typing import List

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class SortItemsVariation1(Task):
    def init_task(self) -> None:
        self._object_1 = Shape("tomato_soup_can")
        self._object_2 = Shape("rubiks_cube")
        self._object_3 = Shape("tennis_ball")
        self._container_detector = ProximitySensor("container_detector")

        self.register_graspable_objects(
            [self._object_1, self._object_2, self._object_3]
        )

        self.register_success_conditions(
            [
                DetectedCondition(self._object_1, self._container_detector),
                DetectedCondition(self._object_2, self._container_detector),
                DetectedCondition(self._object_3, self._container_detector),
            ]
        )

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return ["pick the items and place them into the container"]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1
