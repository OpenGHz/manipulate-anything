from typing import List

from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task


class UseDrawer(Task):
    def init_task(self) -> None:
        # self._options = ["bottom", "middle", "top"]
        self._options = ["middle"]
        self._anchors = [
            Dummy("waypoint_anchor_%s" % opt) for opt in self._options
        ]
        self._joints = [Joint("drawer_joint_%s" % opt) for opt in self._options]
        self._waypoint1 = Dummy("waypoint1")
        self._joints_init_pos = [
            joint.get_joint_position() for joint in self._joints
        ]

    def init_episode(self, index: int) -> List[str]:
        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        for joint, joint_init_pos in zip(self._joints, self._joints_init_pos):
            joint.set_joint_position(joint_init_pos)

        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.15)]
        )
        return [
            "open %s drawer" % option,
            "grip the %s handle and pull the %s drawer open" % (option, option),
            "slide the %s drawer open" % option,
        ]

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self) -> bool:
        return True
