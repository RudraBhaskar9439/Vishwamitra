import gymnasium as gym
import numpy as np

OBS_DIM = 13 # matches SystemState.to_obs_array()

# The meta-agent selects a vector of continuous intervention intensities.
# Each dimension controls ONE category of intervention on [0, 1].
#
# [0]  funding_boost        - inject budget into classrooms
# [1]  teacher_incentive    - bonuses / salary bumps
# [2]  student_scholarship  - merit/need-based scholarships
# [3]  attendance_mandate   - enforcement intensity
# [4]  resource_realloc     - redistribute to under-served schools
# [5]  transparency_report  - public accountability dashboards
# [6]  staff_hiring         - hire new teachers (costly)
# [7]  counseling_programs  - student support services
ACT_DIM = 8

def make_observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.zeros(OBS_DIM, dtype=np.float32),
        high=np.ones(OBS_DIM, dtype=np.float32),
        dtype=np.float32
    )

def make_action_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=np.zeros(ACT_DIM, dtype=np.float32),
        high=np.ones(ACT_DIM, dtype=np.float32),
        dtype=np.float32
    )
