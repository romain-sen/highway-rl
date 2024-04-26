# from highway_env.envs import IntersectionEnv

# Config part two, env racetrack
config2 = {
    # might switch to kinematics if algo don't converge
    "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (64, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.2,
    "controlled_vehicles": 1,
    "other_vehicles": 1, #doesn't work with more than one
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5], # doesn't follow car's direction always the same angle 
    "scaling": 7,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False
}