"""
Example usage of the Sight Reader system.
This demonstrates how to use the SightReader class programmatically.
"""

from sight_reader import SightReader
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def example_basic_usage():
    """Basic example of processing sheet music."""
    print("=== Basic Usage Example ===\n")

    # Initialize the sight reader
    reader = SightReader(template_dir="templates", min_radius=5, max_radius=20)

    # Check if example image exists
    example_images = list(Path("examples").glob("*.jpg")) + list(Path("examples").glob("*.png"))

    if not example_images:
        print("\nNo example images found in examples/ directory")
        print("Please add a sheet music image to the examples/ folder")
        return

    # Process the first example image
    input_image = example_images[0]
    output_image = Path("output") / f"annotated_{input_image.name}"

    print(f"\nProcessing: {input_image}")
    result = reader.process_sheet_music(input_image, output_image)

    print(f"\nDone! Annotated image saved to: {output_image}")

    # Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Detected Notes in {input_image.name}')
    plt.tight_layout()
    plt.show()


def example_batch_processing():
    """Example of processing multiple sheet music images."""
    print("=== Batch Processing Example ===\n")

    reader = SightReader(template_dir="templates", min_radius=5, max_radius=20)

    # Get all images from examples directory
    example_images = list(Path("examples").glob("*.jpg")) + list(Path("examples").glob("*.png"))

    if not example_images:
        print("No images found in examples/ directory")
        return

    print(f"Processing {len(example_images)} images...\n")

    for img_path in example_images:
        print(f"Processing: {img_path.name}")
        output_path = Path("output") / f"annotated_{img_path.name}"

        try:
            reader.process_sheet_music(img_path, output_path)
            print(f"  ✓ Saved to: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nBatch processing complete!")


def example_custom_settings():
    """Example with custom radius settings and visualization."""
    print("=== Custom Settings Example ===\n")

    # Try different radius ranges
    radius_settings = [(3, 15), (5, 20), (8, 25)]
    example_images = list(Path("examples").glob("*.jpg")) + list(Path("examples").glob("*.png"))

    if not example_images:
        print("No images found in examples/ directory")
        return

    input_image = example_images[0]

    fig, axes = plt.subplots(1, len(radius_settings), figsize=(20, 6))

    for idx, (min_r, max_r) in enumerate(radius_settings):
        print(f"Testing with radius range: {min_r}-{max_r}")

        reader = SightReader(template_dir="templates", min_radius=min_r, max_radius=max_r)

        result = reader.process_sheet_music(input_image)

        if len(radius_settings) == 1:
            ax = axes
        else:
            ax = axes[idx]

        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Radius: {min_r}-{max_r}')

    plt.tight_layout()
    plt.show()


def show_all_pitches():
    """Process and display all pitch examples."""
    print("=== All Pitches Viewer ===\n")

    reader = SightReader(template_dir="templates", min_radius=5, max_radius=20)

    # Get all images from examples directory
    example_images = sorted(list(Path("examples").glob("*.jpg")) + list(Path("examples").glob("*.png")))

    if not example_images:
        print("No images found in examples/ directory")
        return

    print(f"Processing {len(example_images)} images...\n")

    num_images = len(example_images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, img_path in enumerate(example_images):
        print(f"Processing: {img_path.name}")
        try:
            result = reader.process_sheet_music(img_path)
            axes[idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[idx].axis('off')
            axes[idx].set_title(img_path.stem)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            axes[idx].axis('off')

    # Hide unused subplots
    if num_images < len(axes):
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Detected Pitches', fontsize=16, y=1.00)
    plt.show()


if __name__ == "__main__":
    import sys

    print("Sight Reading Helper - Example Usage\n")
    print("Choose an example:")
    print("  1. Basic usage (process one image)")
    print("  2. Batch processing (process all images in examples/)")
    print("  3. Custom settings (try different radius ranges)")
    print("  4. View all pitches")
    print()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-4): ")

    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_batch_processing()
    elif choice == "3":
        example_custom_settings()
    elif choice == "4":
        show_all_pitches()
    else:
        print("Invalid choice. Running basic usage example...")
        example_basic_usage()
