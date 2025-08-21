import cv2
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class HomographyApp:
    def __init__(self, display_scale=1.0, num_points=4, ransac_threshold=5.0):
        # Initialize global variables
        self.img1 = None  # Image 1
        self.img2 = None  # Image 2
        self.result = None  # Result image

        # Selected points for each image
        self.img1_points = []
        self.img2_points = []

        # Point selection state
        self.current_img = 1  # 1 for img1, 2 for img2
        self.dragging_point = None  # To track which point is being dragged
        self.dragging_img = None  # To track in which image the point is being dragged

        # Configurable parameters
        self.display_scale = display_scale
        self.num_points = max(4, num_points)  # Minimum 4 points for homography
        self.ransac_threshold = ransac_threshold

        # Homography and inliers
        self.homography_matrix = None
        self.inliers_mask = None

        # Image paths (for file naming)
        self.img1_path = None
        self.img2_path = None

        # Image dimensions (will be defined after loading)
        self.width = 650
        self.height = 1000

        # Create windows with clearer titles
        cv2.namedWindow('Img1 - Source Points')
        cv2.namedWindow('Img2 - Destination Points')
        cv2.namedWindow('Result - Img02 Transformed')
        cv2.namedWindow('Overlay')  # Fixed name to avoid duplicates
        cv2.namedWindow('Homography Matrix', cv2.WINDOW_NORMAL)
        cv2.namedWindow('RANSAC Controls', cv2.WINDOW_NORMAL)

        # Set mouse callbacks
        cv2.setMouseCallback('Img1 - Source Points', self.mouse_callback, 1)
        cv2.setMouseCallback('Img2 - Destination Points', self.mouse_callback, 2)

    def load_images(self, img1_path, img2_path):
        """Load and resize images"""
        # Store paths for file naming
        self.img1_path = img1_path
        self.img2_path = img2_path

        # Load images
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)

        # Check if images are loaded correctly
        if self.img1 is None or self.img2 is None:
            print(f"Error: Unable to load images.")
            print(f"Image 1: {img1_path}")
            print(f"Image 2: {img2_path}")
            return False

        # Get original dimensions
        original_height, original_width = self.img1.shape[:2]

        # Calculate new dimensions with scale factor
        self.width = int(original_width * self.display_scale)
        self.height = int(original_height * self.display_scale)

        # Resize images
        self.img1 = cv2.resize(self.img1, (self.width, self.height))
        self.img2 = cv2.resize(self.img2, (self.width, self.height))

        # Initialize result image as a copy of image 2
        self.result = self.img2.copy()

        # Initialize points with default values
        self.initialize_default_points()

        # Create RANSAC control window
        self.setup_ransac_controls()

        return True

    def setup_ransac_controls(self):
        """Configure RANSAC control window with trackbar"""
        # Trackbar from 1 to 20 directly
        threshold_int = int(self.ransac_threshold)

        # Create trackbar (from 1 to 20)
        cv2.createTrackbar('RANSAC Threshold', 'RANSAC Controls', threshold_int, 20, self.on_threshold_change)

    def on_threshold_change(self, val):
        """Callback for RANSAC threshold change"""
        # Direct trackbar value (no need to divide anymore)
        self.ransac_threshold = max(1, val)  # Minimum 1

        # Immediately recalculate homography with new threshold
        self.compute_homography()

        # Force complete update of all windows
        self.force_refresh_all_windows()

    def force_refresh_all_windows(self):
        """Force refresh of all windows"""
        if self.img1 is not None and self.img2 is not None:
            # Redraw points on images
            img1_display = self.draw_points(self.img1, self.img1_points)
            img2_display = self.draw_points(self.img2, self.img2_points)

            # Create overlay
            overlay_display, alpha, beta = self.create_overlay()

            # Create RANSAC info display
            ransac_info_display = self.create_ransac_info_display()

            # Create and display homography matrix
            matrix_display = self.create_homography_display()

            # Force display of all windows
            cv2.imshow('Img1 - Source Points', img1_display)
            cv2.imshow('Img2 - Destination Points', img2_display)
            cv2.imshow('Result - Img02 Transformed', self.result)
            cv2.imshow('Overlay', overlay_display)
            cv2.imshow('Homography Matrix', matrix_display)
            cv2.imshow('RANSAC Controls', ransac_info_display)

            # Force refresh
            cv2.waitKey(1)

    def create_ransac_info_display(self):
        """Create an image to display RANSAC information"""
        # Reduced info window dimensions
        info_width = 250  # Reduced from 300 to 250
        info_height = 160  # Reduced from 200 to 160

        # Create white image
        info_img = np.ones((info_height, info_width, 3), dtype=np.uint8) * 255

        # Calculate statistics
        total_points = len(self.img1_points)
        inliers_count = 0
        outliers_count = 0

        if self.inliers_mask is not None:
            inliers_count = int(np.sum(self.inliers_mask))
            outliers_count = total_points - inliers_count
        else:
            inliers_count = total_points  # All are inliers if no RANSAC

        # Display information (integer coordinates, smaller font)
        y_offset = 25  # Reduced from 30 to 25
        line_height = 20  # Reduced from 25 to 20

        # Title
        cv2.putText(info_img, "RANSAC Statistics", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Reduced font
        y_offset += int(line_height * 1.5)  # Convert to integer

        # Current threshold
        cv2.putText(info_img, f"Threshold: {self.ransac_threshold:.0f} px", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)  # Reduced font
        y_offset += line_height

        # Total number of points
        cv2.putText(info_img, f"Total Points: {total_points}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)  # Reduced font
        y_offset += line_height

        # Inliers (in green)
        cv2.putText(info_img, f"Inliers: {inliers_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)  # Green and bold, reduced font
        y_offset += line_height

        # Outliers (in red)
        cv2.putText(info_img, f"Outliers: {outliers_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  # Red and bold, reduced font
        y_offset += line_height

        # Inlier percentage
        if total_points > 0:
            inlier_percentage = (inliers_count / total_points) * 100
            cv2.putText(info_img, f"Inlier Rate: {inlier_percentage:.1f}%", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 0), 1)  # Dark green, reduced font

        return info_img

    def save_composite_image(self):
        """Save a composite image with the 4 views and their titles"""
        # Create images with drawn points
        img1_display = self.draw_points(self.img1, self.img1_points)
        img2_display = self.draw_points(self.img2, self.img2_points)
        result_display = self.result.copy()
        overlay_display, alpha, beta = self.create_overlay()

        # Convert from BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1_display, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_display, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB)

        # Create matplotlib figure without main title
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Image 1 - Source Points
        axes[0].imshow(img1_rgb)
        axes[0].set_title('Img1 - Source Points', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Prepare parameters text
        info_text = f"Parameters: {len(self.img1_points)} points, RANSAC threshold: {self.ransac_threshold:.0f}px"
        if self.inliers_mask is not None:
            inliers_count = int(np.sum(self.inliers_mask))
            info_text += f", Inliers: {inliers_count}/{len(self.img1_points)}"

        # Image 2 - Destination Points with parameters in xlabel
        axes[1].imshow(img2_rgb)
        axes[1].set_title('Img2 - Destination Points', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(info_text, fontsize=10, fontweight='normal')
        # Disable ticks but keep labels
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)
        axes[1].spines['left'].set_visible(False)

        # Result - Transformed
        axes[2].imshow(result_rgb)
        axes[2].set_title('Result - Img2 Transformed', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Overlay
        axes[3].imshow(overlay_rgb)
        axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # Adjust layout
        plt.tight_layout()

        # Generate filename based on input images
        import os
        if self.img1_path and self.img2_path:
            # Extract filenames without extension
            img1_name = os.path.splitext(os.path.basename(self.img1_path))[0]
            img2_name = os.path.splitext(os.path.basename(self.img2_path))[0]
            filename = f"output/{img1_name}_{img2_name}_homography_example.png"
        else:
            filename = 'output/homography_composite.png'

        # Save
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()  # Close figure to free memory

        return filename

    def create_homography_display(self):
        """Create an image to display the homography matrix"""
        # Square dimensions for matrix window
        matrix_size = 220  # Reduced and square
        cell_size = matrix_size // 3

        # Create white square image
        matrix_img = np.ones((matrix_size, matrix_size, 3), dtype=np.uint8) * 255

        if self.homography_matrix is not None:
            H = self.homography_matrix

            # Draw grid lines
            for i in range(4):
                # Horizontal lines
                cv2.line(matrix_img, (0, i * cell_size), (matrix_size, i * cell_size), (0, 0, 0), 2)
                # Vertical lines
                cv2.line(matrix_img, (i * cell_size, 0), (i * cell_size, matrix_size), (0, 0, 0), 2)

            # Display matrix values
            for i in range(3):
                for j in range(3):
                    value = H[i, j]

                    # Text position (centered in cell)
                    x = j * cell_size + cell_size // 2
                    y = i * cell_size + cell_size // 2 + 5

                    # Format value with 2 decimal places
                    if abs(value) > 1000 or (abs(value) < 0.01 and value != 0):
                        text = f"{value:.2e}"
                        font_scale = 0.35
                    else:
                        text = f"{value:.2f}"  # 2 decimal places
                        font_scale = 0.4

                    # Calculate text size for centering
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[
                        0]  # thickness=2 for bold
                    text_x = x - text_size[0] // 2
                    text_y = y

                    # Display bold text
                    cv2.putText(matrix_img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)  # thickness=2 for bold

            # Add bold title
            cv2.putText(matrix_img, "Homography Matrix", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  # Smaller title
        else:
            # Display message if no matrix
            cv2.putText(matrix_img, "No Homography", (matrix_size // 2 - 60, matrix_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return matrix_img

    def initialize_default_points(self):
        """Initialize default points according to the number of requested points"""
        if self.num_points == 4:
            # Default points for a rectangle
            margin = 50
            self.img1_points = [
                [margin, margin],
                [self.width - margin, margin],
                [self.width - margin, self.height - margin],
                [margin, self.height - margin]
            ]
            self.img2_points = [
                [margin, margin],
                [self.width - margin, margin],
                [self.width - margin, self.height - margin],
                [margin, self.height - margin]
            ]
        else:
            # Default points for a regular polygon
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(self.width, self.height) // 3

            self.img1_points = []
            self.img2_points = []

            for i in range(self.num_points):
                angle = 2 * np.pi * i / self.num_points
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                self.img1_points.append([x, y])
                self.img2_points.append([x, y])

    def mouse_callback(self, event, x, y, flags, param):
        """Callback for mouse events"""
        img_index = param  # 1 for img1, 2 for img2

        # Get correct point list
        points = self.img1_points if img_index == 1 else self.img2_points

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is close to an existing point
            min_dist = float('inf')
            closest_index = -1

            for i, point in enumerate(points):
                dist = np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i

            # If click is close enough to a point, start dragging that point
            if min_dist < 20:  # 20 pixel threshold
                self.dragging_point = closest_index
                self.dragging_img = img_index

        elif event == cv2.EVENT_MOUSEMOVE:
            # If dragging a point
            if self.dragging_point is not None and self.dragging_img == img_index:
                points[self.dragging_point] = [x, y]
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging point
            if self.dragging_point is not None and self.dragging_img == img_index:
                points[self.dragging_point] = [x, y]
                self.dragging_point = None
                self.dragging_img = None
                self.update_display()

    def draw_points(self, img, points, color_active=(255, 0, 0)):  # Blue for active point
        """Draw points on the image"""
        img_copy = img.copy()

        # Draw points (without lines)
        for i, point in enumerate(points):
            # Determine if point is used (inlier) or not
            is_inlier = True
            if self.inliers_mask is not None and len(self.inliers_mask) == len(points):
                is_inlier = bool(self.inliers_mask[i])

            # Color according to point state
            if self.dragging_point == i:
                color = color_active  # Point being dragged (blue)
                text_color = color_active  # Blue text too
                cv2.circle(img_copy, tuple(point), 5, color, -1)  # Filled
            elif is_inlier:
                color = (0, 255, 0)  # Green for inliers
                text_color = (0, 255, 0)  # Green text
                cv2.circle(img_copy, tuple(point), 5, color, -1)  # Filled
            else:
                color = (0, 0, 255)  # Red for outliers
                text_color = (0, 0, 255)  # Red text
                cv2.circle(img_copy, tuple(point), 5, color, 2)  # Empty

            # Outer circle
            circle_color = (42, 255, 255) if is_inlier else (128, 128, 128)
            cv2.circle(img_copy, tuple(point), 15, circle_color, 1)

            # Add point number with corresponding color
            cv2.putText(img_copy, str(i + 1), (point[0] - 20, point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)  # Point color

        return img_copy

    def create_overlay(self):
        """Create a simple overlay of the result on image 1"""
        if self.result is not None and self.img1 is not None:
            # Option 1: Overlay with normalized transparency
            alpha = 0.5  # Weight for image 1 (base)
            beta = 0.5  # Weight for transformed result (overlay)

            # Ensure alpha + beta = 1.0 to avoid saturation
            total = alpha + beta
            alpha_norm = alpha / total
            beta_norm = beta / total

            # Convert to float to avoid overflow problems
            img1_float = self.img1.astype(np.float32)
            result_float = self.result.astype(np.float32)

            # Normalized blend (not pure addition)
            overlay_float = alpha_norm * img1_float + beta_norm * result_float

            # Convert back to uint8 (no need to clip as normalized)
            overlay = overlay_float.astype(np.uint8)

            return overlay, alpha, beta
        else:
            # Return empty image if no result
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), 1.0, 0.0

    def compute_homography(self):
        """Calculate homography matrix and apply transformation"""
        if len(self.img1_points) < 4 or len(self.img2_points) < 4:
            return

        # Convert points to numpy format
        src_points = np.float32(self.img2_points)  # Source image points (image 2)
        dst_points = np.float32(self.img1_points)  # Destination image points (image 1)

        # Calculate homography matrix
        if len(src_points) == 4:
            H, _ = cv2.findHomography(src_points, dst_points)
            self.inliers_mask = np.ones(4, dtype=bool)  # All points are used with 4 points
        else:
            # For more than 4 points, use RANSAC method
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,
                                         ransacReprojThreshold=self.ransac_threshold)
            self.inliers_mask = mask.flatten().astype(bool) if mask is not None else None

        self.homography_matrix = H

        if H is not None:
            print('Homography Matrix:')
            print(H)
            if self.inliers_mask is not None and len(self.inliers_mask) > 4:
                inlier_count = np.sum(self.inliers_mask)
                print(f'RANSAC threshold: {self.ransac_threshold} pixels')
                print(f'Inliers: {inlier_count}/{len(self.inliers_mask)} points')
                print(f'Inlier indices: {np.where(self.inliers_mask)[0] + 1}')  # +1 for human-friendly display
            print()

            # Apply perspective transformation
            self.result = cv2.warpPerspective(self.img2, H, (self.width, self.height))
        else:
            print("Unable to calculate homography matrix")
            self.inliers_mask = None

    def update_display(self):
        """Update display of images with points and result"""
        # Draw points on images
        img1_display = self.draw_points(self.img1, self.img1_points)
        img2_display = self.draw_points(self.img2, self.img2_points)

        # Calculate and apply homography
        self.compute_homography()

        # Create overlay and get alpha/beta values
        overlay_display, alpha, beta = self.create_overlay()

        # Create and display homography matrix
        matrix_display = self.create_homography_display()

        # Create RANSAC info display
        ransac_info_display = self.create_ransac_info_display()

        # Display images with fixed titles
        cv2.imshow('Img1 - Source Points', img1_display)
        cv2.imshow('Img2 - Destination Points', img2_display)
        cv2.imshow('Result - Img02 Transformed', self.result)
        cv2.imshow('Overlay', overlay_display)
        cv2.imshow('Homography Matrix', matrix_display)
        cv2.imshow('RANSAC Controls', ransac_info_display)

    def run(self):
        """Main application loop"""
        # Display initial images
        self.update_display()

        print("Instructions:")
        print(f"- {self.num_points} points to place on each image")
        print("- Place points on Img1 (source) to delimit the area to transform")
        print("- Place points on Img2 (destination) to delimit the target area")
        print("- Click and drag points to move them")
        print("- GREEN numbering = points used by RANSAC (inliers)")
        print("- RED numbering = points not used by RANSAC (outliers)")
        print(f"- Initial RANSAC threshold: {self.ransac_threshold} pixels")
        print("- Use the 'RANSAC Threshold' slider (1-20) to adjust threshold")
        print("- The 'RANSAC Controls' window shows real-time statistics")
        print("- The homography matrix is displayed in a square window")
        print("- The overlay uses normalized superposition (avoids saturation)")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset points")
        print("- Press 's' to save a composite image with the 4 views")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset points
                self.initialize_default_points()
                self.update_display()
                print("Points reset")
            elif key == ord('s'):
                # Save composite image
                try:
                    filename = self.save_composite_image()
                    print(f"Composite image saved to '{filename}'")
                    print(f"Parameters used:")
                    print(f"  - Points: {len(self.img1_points)}")
                    print(f"  - RANSAC threshold: {self.ransac_threshold:.0f} pixels")
                    if self.inliers_mask is not None:
                        inliers_count = int(np.sum(self.inliers_mask))
                        print(
                            f"  - Inliers: {inliers_count}/{len(self.img1_points)} ({inliers_count / len(self.img1_points) * 100:.1f}%)")
                except Exception as e:
                    print(f"Error during save: {e}")
                    # Fallback to original method
                    cv2.imwrite('homography_result.jpg', self.result)
                    overlay, alpha, beta = self.create_overlay()
                    cv2.imwrite('homography_overlay.jpg', overlay)
                    print("Images saved individually as fallback")

        cv2.destroyAllWindows()


def main():
    """Main function with argument handling"""
    parser = argparse.ArgumentParser(description='Interactive homography application')
    parser.add_argument('img1', help='Path to the first image (reference)')
    parser.add_argument('img2', help='Path to the second image (source)')
    parser.add_argument('--display-scale', '-s', type=float, default=1.0,
                        help='Display scale factor (0.1 to 1.0, default: 1.0)')
    parser.add_argument('--num-points', '-n', type=int, default=4,
                        help='Number of points to use (minimum 4, default: 4)')
    parser.add_argument('--ransac-threshold', '-r', type=float, default=5.0,
                        help='RANSAC reprojection threshold in pixels (1-20, default: 5.0)')

    args = parser.parse_args()

    # Validate arguments
    if args.display_scale <= 0 or args.display_scale > 1:
        print("Error: display-scale must be between 0.1 and 1.0")
        sys.exit(1)

    if args.num_points < 4:
        print("Error: num-points must be at least 4")
        sys.exit(1)

    if args.ransac_threshold <= 0 or args.ransac_threshold > 20:
        print("Error: ransac-threshold must be between 1 and 20")
        sys.exit(1)

    # Create and launch application
    app = HomographyApp(display_scale=args.display_scale,
                        num_points=args.num_points,
                        ransac_threshold=args.ransac_threshold)

    success = app.load_images(args.img1, args.img2)

    if success:
        print(f"Images loaded successfully")
        print(f"Display scale: {args.display_scale}")
        print(f"Number of points: {args.num_points}")
        print(f"RANSAC threshold: {args.ransac_threshold:.0f} pixels")
        print(f"Dimensions: {app.width}x{app.height}")
        print()
        app.run()
    else:
        print("Unable to launch application. Check image paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()