import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse

def create_comparison_figure(base_path, iterations, test_images, output_name='rendering_comparison.png'):
    """
    Create a comparison figure for NeRF training progress
    
    Parameters:
    -----------
    base_path : str or Path
        Base path to the images, e.g., 'logs/blender_lego_fine'
    iterations : list of int
        List of iteration numbers to display, e.g., [500, 2000, 5000, 10000, 20000]
    test_images : list of int
        List of test image indices, e.g., [0, 1, 2]
    output_name : str
        Output filename
    """
    
    base_path = Path(base_path)
    
    # Create figure
    n_rows = len(test_images) * 2
    n_cols = len(iterations)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 16))
    
    # Adjust subplot spacing
    plt.subplots_adjust(left=0.08, right=0.95, wspace=0.05, hspace=0.15, top=0.95, bottom=0.05)
    
    # Iterate over each test image
    for test_idx in test_images:
        row_rgb = test_idx * 2
        row_depth = test_idx * 2 + 1
        
        # Iterate over each iteration number
        for col_idx, iter_num in enumerate(iterations):
            # Build file paths
            folder_name = f'testset_{iter_num:06d}'
            rgb_path = base_path / folder_name / f'{test_idx:03d}_rgb.png'
            depth_path = base_path / folder_name / f'{test_idx:03d}_depth.png'
            
            # Display RGB image
            try:
                rgb_img = mpimg.imread(rgb_path)
                axes[row_rgb, col_idx].imshow(rgb_img)
                axes[row_rgb, col_idx].axis('off')
                
                # Add title only to the first row
                if test_idx == test_images[0]:
                    axes[row_rgb, col_idx].set_title(
                        f'iter_{iter_num}',
                        fontsize=14,
                        pad=10
                    )
            except FileNotFoundError:
                axes[row_rgb, col_idx].text(
                    0.5, 0.5, 'Image\nNot Found',
                    ha='center', va='center',
                    fontsize=12, color='red'
                )
                axes[row_rgb, col_idx].axis('off')
                print(f"Warning: File not found {rgb_path}")
            
            # Display Depth image
            try:
                depth_img = mpimg.imread(depth_path)
                im = axes[row_depth, col_idx].imshow(depth_img, cmap='viridis')
                axes[row_depth, col_idx].axis('off')
                
            except FileNotFoundError:
                axes[row_depth, col_idx].text(
                    0.5, 0.5, 'Depth\nNot Found',
                    ha='center', va='center',
                    fontsize=12, color='red'
                )
                axes[row_depth, col_idx].axis('off')
                print(f"Warning: File not found {depth_path}")
    
    # Redraw to get correct coordinates
    fig.canvas.draw()
    
    # Add test image labels on the left side of each group
    for i, test_idx in enumerate(test_images):
        row_rgb = i * 2
        row_depth = i * 2 + 1
        
        # Get actual positions of the two subplots
        bbox_rgb = axes[row_rgb, 0].get_position()
        bbox_depth = axes[row_depth, 0].get_position()
        
        # Calculate middle y-coordinate between the two rows
        y_middle = (bbox_rgb.y0 + bbox_depth.y1) / 2
        
        fig.text(
            0.02,
            y_middle,
            f'Test image {test_idx}',
            fontsize=14,
            fontweight='bold',
            va='center',
            ha='left',
            transform=fig.transFigure
        )
    
    # Add main title
    fig.text(
        0.5, 0.01,
        'NeRF test-view renderings (RGB and depth) across training iterations',
        fontsize=16,
        fontweight='bold',
        ha='center',
        va='bottom'
    )
    
    # Save figure
    plt.savefig(
        output_name,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"✓ Figure saved as: {output_name}")
    
    plt.show()


if __name__ == "__main__":
    # Method 1: Use command line arguments
    parser = argparse.ArgumentParser(description='Generate NeRF rendering comparison figure')
    parser.add_argument('--base_path', type=str, default='logs/blender_lego_fine',
                        help='Base path to the images')
    parser.add_argument('--iterations', type=int, nargs='+', 
                        default=[500, 2000, 5000, 10000, 20000],
                        help='List of iteration numbers')
    parser.add_argument('--test_images', type=int, nargs='+',
                        default=[0, 1, 2],
                        help='List of test image indices')
    parser.add_argument('--output', type=str, default='rendering_comparison.png',
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Execute
    create_comparison_figure(
        base_path=args.base_path,
        iterations=args.iterations,
        test_images=args.test_images,
        output_name=args.output
    )