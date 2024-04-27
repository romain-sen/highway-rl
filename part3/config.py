from highway_env.envs import IntersectionEnv


# Config part three, env intersection
config = {
    # might switch to kinematics if algo don't converge
    "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (64, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True
        },
    "duration": 13,  # [s]
    "destination": "o1", #hardest destination
    "initial_vehicle_count": 12,
    "spawn_probability": 0.7,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 5.5 * 1.3,
    "collision_reward": IntersectionEnv.default_config()['collision_reward'],
    "normalize_reward": False,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}