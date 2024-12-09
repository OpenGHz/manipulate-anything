from typing import List
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import (
    DetectedSeveralCondition, FalseCondition, NothingGrasped
) 
from rlbench.backend.task import Task


class ChangeClock(Task):

    def init_task(self) -> None:
        self.register_success_conditions([FalseCondition()])
        self.sponge = Shape('clock_needle_crank')
        self.register_graspable_objects([self.sponge])

    def init_episode(self, index: int) -> List[str]:
        return [
            'change the clock to show time 12.15',
            'adjust the time to 12.15',
            'change the clock to 12.15',
            'set the clock to 12.15',
            'turn the knob on the back of the clock until the time shows 12.15',
            'rotate the wheel on the clock to make it show 12.15',
            'make the clock say 12.15',
            'turn the knob on the back of the clock 90 degrees'
        ]

    def variation_count(self) -> int:
        return 1
