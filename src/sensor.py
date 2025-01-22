import numpy as np
from scipy.ndimage import gaussian_filter

class Sensor:
    def __init__(self, points, colors):
        self.point_locs = points
        self.point_cols = colors
        
    # Function to compute rotation matrix for X, Y, and Z axes given the camera position and a target position
    def compute_rotation_matrix(self, camera_position, target_position, up_vector=np.array([0, 1, 0])):
        # Compute the forward vector (from camera to target)
        forward = np.array(target_position) - np.array(camera_position)
        forward = forward / np.linalg.norm(forward)
        
        # Compute the right vector (orthogonal to forward and up)
        right = np.cross(up_vector, forward)
        right = right / np.linalg.norm(right)
        
        # Compute the up vector (orthogonal to forward and right)
        up = np.cross(forward, right)
        
        # Create the rotation matrix
        rotation_matrix = np.array([right, up, forward])
        
        return rotation_matrix
    
    # Helper function to apply the pinhole camera model for projection with rotation
    def pinhole_projection_helper(self, camera_position, camera_rotation, focal_length=1.0):
        # Shift points based on camera position
        shifted_points = self.point_locs - camera_position
    
        # Apply camera rotation to the points
        rotated_points = np.dot(shifted_points, camera_rotation.T)
    
        # Camera forward vector (assumes it's the negative Z-axis in camera space)
        camera_forward = np.array([0, 0, 1])
    
        # Check dot product to determine points "behind" the camera
        dot_products = np.dot(rotated_points, camera_forward)
        valid_indices = dot_products > 0  # Points in front of the camera
    
        # Filter out points behind the camera
        rotated_points = rotated_points[valid_indices]
        shifted_points = shifted_points[valid_indices]
    
        # Project points using pinhole camera model
        X = -rotated_points[:, 0]
        Y = rotated_points[:, 1]
        Z = rotated_points[:, 2]
    
        # Avoid division by zero for points too close to the camera
        Z[Z == 0] = 1e-6
    
        # Apply the projection to get 2D coordinates (pinhole projection formula)
        x_proj = focal_length * X / Z
        y_proj = focal_length * Y / Z
    
        # Get magnitude (distance from point to camera)
        mags = np.linalg.norm(shifted_points, axis=1)
    
        # Return 2D projected points and distances for valid points
        return np.vstack((x_proj, y_proj)).T, self.point_cols[valid_indices], mags

    def pinhole_project_target(self, camera_position, target_position, focal_length=1.0, up_vector=np.array([0, 1, 0])):
        # Create the camera rotation matrix
        camera_rotation = self.compute_rotation_matrix(camera_position, target_position, up_vector)

        target_shifted = np.array(target_position) - camera_position
        target_rotated = np.dot(target_shifted, camera_rotation.T)

        # Project target using pinhole camera model
        X = -target_rotated[0]
        Y = target_rotated[1]
        Z = target_rotated[2]
        
        x_proj = focal_length * X / Z
        y_proj = focal_length * Y / Z
    
        # Get magnitude (distance from target point to camera)
        mag = np.linalg.norm(target_shifted)
    
        return np.array((x_proj, y_proj)), mag

    def pinhole_projection(self, camera_position, target_position, focal_length=1.0, up_vector=np.array([0, 1, 0])):
        # Create the camera rotation matrix
        camera_rotation = self.compute_rotation_matrix(camera_position, target_position, up_vector)
       
        # Project the 3D points onto the 2D plane using pinhole model
        projected_points, colors, dists = self.pinhole_projection_helper(camera_position, camera_rotation, focal_length)
       
        return projected_points, colors, dists
   
    def pinhole_projection_noisy(self, camera_position, target_position, focal_length=1.0, up_vector=np.array([0, 1, 0]), propagation_factor=0.04):       
        # Project the 3D points onto the 2D plane using pinhole model
        projected_points, colors, dists = self.pinhole_projection(camera_position, target_position, focal_length, up_vector)
       
        # Add propagation error (error increases with distance)
        noisy_dists = dists + np.random.normal(0, propagation_factor * dists, size=dists.shape)

        return projected_points, colors, noisy_dists
   
    def pinhole_projection_blur(self, camera_position, target_position, focal_length=1.0, up_vector=np.array([0, 1, 0]), propagation_factor=0.04):
        # Obtainn projected points and noisy dist
        projected_points, colors, noisy_dists = self.pinhole_projection_noisy(camera_position, target_position, focal_length, up_vector, propagation_factor)
        
        # Blur only far-away points
        blurred_noisy_dists = gaussian_filter(noisy_dists, sigma=5)

        # Return the projected points and the blurred noisy distances
        return projected_points, colors, blurred_noisy_dists

    def is_target_visible(self, camera_position, target_position, similarity_thresh=1, propagation_factor=0.04):
        # Setting camera position as origin
        shifted_points = self.point_locs - camera_position
        target_position = np.array(target_position) - camera_position

        point_dists = np.linalg.norm(shifted_points, axis=1, keepdims=True)
        target_dist = np.linalg.norm(target_position)
        
        # Normalize vectors
        target_dir = target_position / target_dist
        shifted_points_dir = shifted_points / point_dists
        point_dists = point_dists.flatten() 
        point_dists += np.random.normal(0, propagation_factor * point_dists, size=point_dists.shape)
        
        # Calculate cosine similarity
        cosine_similarities = np.dot(shifted_points_dir, target_dir)
        
        # Check if any point satisfies the conditions
        closer_and_similar = (cosine_similarities >= similarity_thresh) & (point_dists <= target_dist)

        return not np.any(closer_and_similar)