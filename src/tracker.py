import numpy as np
from tqdm import tqdm

class Tracker:
    def __init__(self, sensor, step_size, propagation_factor, add_noise, apply_blur, height_adjustment):
        self.sensor = sensor
        self.step_size = step_size
        self.add_noise = add_noise
        self.propagation_factor = propagation_factor
        self.apply_blur = apply_blur
        self.height_adjustment = height_adjustment

    def track(self, target_path, initial_tracker_pos):        
        current_tracker_pos = initial_tracker_pos
        target_last_obs_loc = None
        tracker_path = []
        tracker_view_target_path = []

        # Simulate target path and tracker  path
        for i, target_loc in tqdm(enumerate(target_path)):
        
            # Add noise to the target position if enabled
            if self.add_noise:
                dist = np.linalg.norm(target_loc - current_tracker_pos)
                target_loc += np.random.normal(0, self.propagation_factor*dist, size=target_loc.size)*(target_loc - current_tracker_pos)/dist
        
            # Apply blur to the target position if enabled
            adjusted_target_loc = target_loc
            if self.apply_blur:
                recent_positions = target_path[max(0, i - 4): i+1]
                adjusted_target_loc = recent_positions.mean(0)
            tracker_view_target_path.append(adjusted_target_loc)
                
            if self.sensor.is_target_visible(current_tracker_pos, adjusted_target_loc, propagation_factor=self.propagation_factor):
                # Take step toward adjusted_target_loc
                step_dir = adjusted_target_loc - current_tracker_pos
                target_last_obs_loc = adjusted_target_loc
                new_height = adjusted_target_loc[2] + self.height_adjustment
            else:
                # Take step toward target_last_obs_loc 
                step_dir = target_last_obs_loc-current_tracker_pos
                new_height = target_last_obs_loc[2] + self.height_adjustment
        
            step_dir /= np.linalg.norm(step_dir)
            step = step_dir * self.step_size
        
            new_tracker_pos = current_tracker_pos + step
            new_tracker_pos[2] = new_height
            
            current_tracker_pos = new_tracker_pos
            tracker_path.append(new_tracker_pos)

        return tracker_path, tracker_view_target_path