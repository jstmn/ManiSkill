from .open_drawer import OpenDrawerEnv
from .raise_cube import RaiseCubeEnv
from .peg_insertion_side_v2 import PegInsertionSideColosseumV2
from .dual_panda_stack_cube import TwoRobotStackCube
from .place_dish_in_rack import PlaceDishInRackEnv
from .pick_dish_from_rack import PickDishFromRackEnv
from .cook_item_in_pan import CookItemInPanEnv
from .open_cabinet import OpenCabinetEnv
from .scoop_banana import ScoopBananaEnv
from .hammer_nail import HammerNailEnv
from .dual_panda_pick_cube import DualArmPickCubeEnv
from .dual_panda_drawer_place import DualArmDrawerPlaceEnv
from .dual_panda_drawer_open import DualArmDrawerOpenEnv
from .dual_panda_pick_bottle import DualArmPickBottleEnv
from .dual_panda_pick_pour_pot import DualArmPourPotEnv
from .dual_panda_lift_pot import DualArmLiftPotEnv
from .dual_panda_lift_tray import DualArmLiftTrayEnv
from .dual_panda_threading import DualPandaThreadingEnv
from .dual_panda_stack3cubes import TwoRobotStack3Cube
from .dual_panda_pen_cap import DualArmPenCapEnv
from .dual_panda_push_box import DualPandaPushBoxEnv
from .place_cube_in_drawer import PlaceCubeInDrawerEnv
from .pick_soda_from_cabinet import PickSodaFromCabinetEnv
from .book_in_shelf import PlaceBookEnv
from .rotate_arrow import RotateArrowEnv
from .stack_cube_cv2 import StackCubeColosseumV2Env
from .lift_peg_upright_cv2 import LiftPegUprightColosseumV2Env
from .plug_charger_v2 import PlugChargerColosseumV2Env


"""
# Data from 'scripts/data_generation/motionplanning_colosseum_v2_single_arm.sh'
================================================================================
Env ID                              | Count      | Mean Episode Length | Std Episode Length | Max Episode Length
--------------------------------------------------------------------------------
PickSodaFromCabinet-v1              | 96         | 193.57     | 2.5        | 198.0
PickDishFromRack-v1                 | 100        | 119.76     | 9.26       | 134.0
StackCubeColosseumV2-v1             | 100        | 107.14     | 8.97       | 130.0
PlaceDishInRack-v1                  | 70         | 251.36     | 19.07      | 322.0
LiftPegUprightColosseumV2-v1        | 97         | 198.42     | 6.51       | 225.0
RotateArrow-v1                      | 100        | 328.94     | 6.94       | 351.0
PegInsertionSideColosseumV2-v1      | 96         | 151.35     | 28.91      | 203.0
PlugChargerColosseumV2-v1           | 96         | 179.6      | 13.83      | 208.0
HammerNail-v1                       | 96         | 225.73     | 5.6        | 243.0
ScoopBanana-v1                      | 99         | 242.9      | 20.05      | 375.0
OpenDrawer-v1                       | 100        | 118.36     | 3.72       | 126.0
OpenCabinet-v1                      | 93         | 475.9      | 4.15       | 483.0
PlaceCubeInDrawer-v1                | 96         | 333.45     | 13.62      | 373.0
PlaceBookInShelf-v1                 | 100        | 182.5      | 8.54       | 202.0
CookItemInPan-v1                    | 100        | 473.93     | 15.12      | 560.0
RaiseCube-v1                        | 100        | 78.25      | 3.55       | 86.0
--------------------------------------------------------------------------------
Total Merged Episodes               | 1539      
================================================================================

# Data from scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh
================================================================================
Env ID                              | Count      | Mean Episode Length | Std Episode Length | Max Episode Length
--------------------------------------------------------------------------------
DualArmPickCube-v1                  | 98         | 201.1      | 3.2        | 211.0
DualArmPickBottle-v1                | 98         | 130.72     | 6.23       | 154.0
DualArmLiftPot-v1                   | 98         | 98.06      | 6.94       | 112.0
DualArmLiftTray-v1                  | 98         | 104.72     | 4.43       | 117.0
DualArmPushBox-v1                   | 98         | 93.04      | 9.43       | 113.0
DualArmPourPot-v1                   | 98         | 200.72     | 3.5        | 209.0
DualArmThreading-v1                 | 100        | 164.97     | 6.92       | 182.0
DualArmPenCap-v1                    | 100        | 186.1      | 11.54      | 230.0
DualArmDrawerPlace-v1               | 100        | 186.35     | 4.0        | 195.0
DualArmDrawerOpen-v1                | 100        | 81.0       | 9.4        | 100.0
DualArmStackCube-v1                 | 100        | 137.03     | 7.27       | 154.0
DualArmStack3Cube-v1                | 100        | 242.08     | 10.5       | 260.0
--------------------------------------------------------------------------------
Total Merged Episodes               | 1188      
================================================================================
"""

# Set to mean + 4*std
MAX_EPISODE_STEPS_BY_TASK = {
    "PickSodaFromCabinet-v1": int(193 + 4*2.5),
    "PickDishFromRack-v1": int(119 + 4*9.26),
    "StackCubeColosseumV2-v1": int(107 + 4*8.97),
    "PlaceDishInRack-v1": int(251 + 4*19.07),
    "LiftPegUprightColosseumV2-v1": int(198 + 4*6.51),
    "RotateArrow-v1": int(328 + 4*6.94),
    "PegInsertionSideColosseumV2-v1": int(151 + 4*28.91),
    "PlugChargerColosseumV2-v1": int(179 + 4*13.83),
    "HammerNail-v1": int(225 + 4*5.6),
    "ScoopBanana-v1": int(242 + 4*20.05),
    "OpenDrawer-v1": int(118 + 4*3.72),
    "OpenCabinet-v1": int(475 + 4*4.15),
    "PlaceCubeInDrawer-v1": int(333 + 4*13.62),
    "PlaceBookInShelf-v1": int(182 + 4*8.54),
    "CookItemInPan-v1": int(473 + 4*15.12),
    "RaiseCube-v1": int(78 + 4*3.55),
    "DualArmPickCube-v1": int(201.1 + 4*3.2),
    "DualArmPickBottle-v1": int(130.72 + 4*6.23),
    "DualArmLiftPot-v1": int(98.06 + 4*6.94),
    "DualArmLiftTray-v1": int(104.72 + 4*4.43),
    "DualArmPushBox-v1": int(93.04 + 4*9.43),
    "DualArmPourPot-v1": int(200.72 + 4*3.5),
    "DualArmThreading-v1": int(164.97 + 4*6.92),
    "DualArmPenCap-v1": int(186.1 + 4*11.54),
    "DualArmDrawerPlace-v1": int(186.35 + 4*4.0),
    "DualArmDrawerOpen-v1": int(81.0 + 4*9.4),
    "DualArmStackCube-v1": int(137.03 + 4*7.27),
    "DualArmStack3Cube-v1": int(242.08 + 4*10.5),
}
