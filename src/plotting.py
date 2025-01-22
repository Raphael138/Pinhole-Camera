import numpy as np
import matplotlib.pyplot as plt
from src.sensor import Sensor

def render_pinhole_projection(sensor, camera_position, target_position, focal_length=1.0):
    """
    Renders a pinhole projection of a 3D cloud onto a 2D plane.
    
    Parameters:
    - sensor: Sensor object initiated with point cloud.
    - camera_position: The 3D position of the camera.
    - target_position: The target point the camera is looking at.
    - focal_length: The focal length of the pinhole camera.

    Returns:
    - fig, ax: Matplotlib figure and axis with the rendered projection.
    """

    # Create fig/axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create the camera rotation matrix
    up_vector = np.array([0, 0, 1])
    projected_points, colors, _ = sensor.pinhole_projection(camera_position, target_position, focal_length, up_vector)
        
    # Plot the 2D points
    ax.scatter(projected_points[:, 0], projected_points[:, 1], c=colors, s=5, alpha=0.6)
    ax.set_xlabel("x'")
    ax.set_ylabel("y'")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.1, 0.2)
    ax.set_xlim(-0.8, 0.3)
    ax.set_title('Pinhole Camera')
    
    return fig, ax

def render_pixelated_pinhole_projection(sensor, camera_position, target_position, focal_length=1.0, grid_size=(300, 300)):
    """
    Renders a pixelated version of the pinhole projection of a 3D scene.
    
    Parameters:
    - sensor: Sensor object initiated with point cloud.
    - camera_position: The 3D position of the camera.
    - target_position: The target point the camera is looking at.
    - focal_length: The focal length of the pinhole camera.
    - grid_size: The resolution of the pixelated output (height, width).

    Returns:
    - None. Displays the pixelated projection.
    """

    # Project the 3D points onto the 2D plane using pinhole model
    up_vector = np.array([0, 0, 1])
    projected_points, colors, dists = sensor.pinhole_projection(camera_position, target_position, focal_length, up_vector)

    # Create a blank color image for pixelated view
    pixelated_image = np.full((*grid_size, 3), fill_value=255)  # White background

    # Create a distance matrix to track closest points
    pixel_distances = np.full(grid_size, np.inf)

    # Map projected points to pixel locations
    min_x, max_x, min_y, max_y = -.8, .3, -.1, 0.2
    x_pixels = ((projected_points[:,0]-min_x) * grid_size[1]/(max_x-min_x)).astype(np.int64)
    y_pixels = ((max_y-projected_points[:,1]) * grid_size[0]/(max_y-min_y)).astype(np.int64)

    indices = (0 <= x_pixels) & (x_pixels < grid_size[1]) & (0 <= y_pixels) & (y_pixels < grid_size[0])
    indices = np.where(indices)[0] # Sub-selection of points which are in the projection screen

    for i in indices:
        x_pixel, y_pixel, col, dist = x_pixels[i], y_pixels[i], colors[i], dists[i]

        if dist < pixel_distances[y_pixel, x_pixel]: # Check if this point is closer
            
            pixel_distances[y_pixel, x_pixel] = dist
            pixelated_image[y_pixel, x_pixel] = col*255  # Use the color of the nearest point
                
    # Plot the pixelated image
    plt.figure(figsize=(8, 8))
    plt.imshow(pixelated_image.astype(np.uint8))
    plt.axis('off')  # Hide axis values
    plt.title('Pixelated Pinhole Camera Model')
    plt.show()

def render_depth_sensor(sensor, camera_position, target_position, focal_length=1.0, grid_size=(200, 200), sensor_type="ideal", plot_target= False):
    """
    Renders a depth sensor projection based on camera position, target position, and focal length
    using a pinhole model.

    Parameters:
    - sensor (object): The sensor object.
    - camera_position (tuple): The 3D coordinates of the camera.
    - target_position (tuple): The 3D coordinates of the target object in the scene.
    - focal_length (float, optional): The focal length of the camera. Default is 1.0.
    - grid_size (tuple, optional): The resolution of the output image (height, width). Default is (200, 200).
    - sensor_type (str, optional): Specifies the sensor type for the projection. Options are:
      - "ideal" for perfect projection
      - "noisy" for projections with noise
      - "blurry" for projections with blur

    Returns:
    - None: The function displays the generated depth map image.
    """
    # Create fig/axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define the up vector for the camera, which is used to rotate the projection
    up_vector = np.array([0, 0, 1])

    if sensor_type == "ideal":
        # Ideal pinhole projection with no noise or blur
        projected_points, colors, dists = sensor.pinhole_projection(camera_position, target_position, focal_length, up_vector)
    elif sensor_type == "noisy":
        # Noisy pinhole projection with added noise
        projected_points, colors, dists = sensor.pinhole_projection_noisy(camera_position, target_position, focal_length, up_vector)
    elif sensor_type == "blurry":
        # Blurry pinhole projection with added blur
        projected_points, colors, dists = sensor.pinhole_projection_blur(camera_position, target_position, focal_length, up_vector)

    # Create an empty image to store the depth values (pixelated view)
    pixelated_image = np.full(grid_size, fill_value=np.inf)

    # Define the projection boundaries in normalized space
    min_x, max_x, min_y, max_y = -.8, .3, -.1, .2

    # x_pixels and y_pixels map the projected points to the image grid
    x_pixels = ((projected_points[:, 0] - min_x) * grid_size[1] / (max_x - min_x)).astype(np.int64)
    y_pixels = ((projected_points[:, 1] - min_y) * grid_size[0] / (max_y - min_y)).astype(np.int64)

    # Select the points that fall within the bounds of the image grid
    indices = (0 <= x_pixels) & (x_pixels < grid_size[1]) & (0 <= y_pixels) & (y_pixels < grid_size[0])
    indices = np.where(indices)[0]

    # Iterate over the valid indices to update the pixelated image with the closest depth values
    for i in indices:
        x_pixel, y_pixel, dist = x_pixels[i], y_pixels[i], dists[i]

        # Only update the pixel if the current point is closer (smaller depth)
        if dist < pixelated_image[y_pixel, x_pixel]:
            pixelated_image[y_pixel, x_pixel] = dist    

    # Mask the infinite values in the depth image (for invalid or unassigned pixels)
    masked_image = np.ma.masked_invalid(pixelated_image)

    # Set up the colormap to visualize the depth values
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')

    # Plot target if needed
    if plot_target:
        s = Sensor(None, None)
        target_xy, target_mag = s.pinhole_project_target(camera_position, target_position, focal_length=1.0, up_vector=up_vector)
        x_pixel = int((target_xy[0] - min_x) * grid_size[1] / (max_x - min_x)) 
        y_pixel = int((target_xy[1] - min_y) * grid_size[0] / (max_y - min_y))
        draw_target(masked_image, x_pixel, y_pixel, masked_image.max())

    # Plot the pixelated depth image using the colormap
    ax.imshow(-masked_image, cmap=cmap, interpolation='nearest', origin='lower')
    ax.axis('off')
    ax.set_title('Depth Sensor')

    return fig, ax
    
def draw_target(mat, pixel_x, pixel_y, color, size=3):
    """
    Draws a cross (including diagonal lines) centered at a specified position 
    in a 2D matrix with a given size and color.

    Parameters:
    - mat (numpy.ndarray): The matrix where the target will be drawn. Can be 2D or 3D (e.g., RGB).
    - pixel_x (int): The x-coordinate (column) of the target's center.
    - pixel_y (int): The y-coordinate (row) of the target's center.
    - color (int or tuple): The value to set at the target's pixels. If the matrix is grayscale, 
                            this should be an integer; for RGB matrices, this should be a tuple of values.
    - size (int, optional): The "radius" of the cross. Larger values produce larger crosses. Default is 3.

    Returns:
    - None: The matrix `mat` is updated directly.
    """

    # Ensure coordinates are within the bounds of the matrix
    rows, cols = mat.shape[:2]

    for offset in range(-size, size + 1):
        # Vertical line
        if 0 <= pixel_y + offset < rows:
            if 0 <= pixel_x < cols:
                mat[pixel_y + offset, pixel_x] = color

        # Horizontal line
        if 0 <= pixel_x + offset < cols:
            if 0 <= pixel_y < rows:
                mat[pixel_y, pixel_x + offset] = color

        # Diagonal lines
        if 0 <= pixel_y + offset < rows and 0 <= pixel_x + offset < cols:
            mat[pixel_y + offset, pixel_x + offset] = color  # Top-left to bottom-right
        if 0 <= pixel_y + offset < rows and 0 <= pixel_x - offset < cols:
            mat[pixel_y + offset, pixel_x - offset] = color  # Top-right to bottom-left