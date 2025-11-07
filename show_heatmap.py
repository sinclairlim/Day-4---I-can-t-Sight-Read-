"""
Visualize template matching heatmaps for the sight reading system.
Shows how the system finds noteheads using find.png template.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_heatmap_example(image_path, template_path, scale=1.0):
    """
    Show a heatmap visualization of template matching.

    Args:
        image_path: Path to sheet music image
        template_path: Path to template (e.g., find.png)
        scale: Scale factor for the template
    """
    # Load images
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    if image is None or template is None:
        print("Error loading images!")
        return

    # Resize template
    template_h, template_w = template.shape
    scaled_w = int(template_w * scale)
    scaled_h = int(template_h * scale)
    resized_template = cv2.resize(template, (scaled_w, scaled_h))

    # Perform template matching
    result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

    # Find the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Sheet Music')
    axes[0, 0].axis('off')

    # 2. Template
    axes[0, 1].imshow(resized_template, cmap='gray')
    axes[0, 1].set_title(f'Template (scale={scale}x)\nSize: {scaled_w}x{scaled_h}')
    axes[0, 1].axis('off')

    # 3. Heatmap
    heatmap = axes[0, 2].imshow(result, cmap='jet', interpolation='nearest')
    axes[0, 2].set_title(f'Similarity Heatmap\nMax: {max_val:.3f}, Min: {min_val:.3f}')
    axes[0, 2].axis('off')
    plt.colorbar(heatmap, ax=axes[0, 2])

    # 4. Heatmap thresholded at 0.5
    threshold = 0.5
    result_thresh = result.copy()
    result_thresh[result_thresh < threshold] = 0
    heatmap_thresh = axes[1, 0].imshow(result_thresh, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title(f'Heatmap (threshold >= {threshold})')
    axes[1, 0].axis('off')
    plt.colorbar(heatmap_thresh, ax=axes[1, 0])

    # 5. Detections on original image
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        x, y = pt
        cv2.rectangle(output, pt, (pt[0] + scaled_w, pt[1] + scaled_h), (0, 255, 0), 2)

    axes[1, 1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Detections (threshold >= {threshold})\nFound: {len(locations[0])} matches')
    axes[1, 1].axis('off')

    # 6. Best match highlighted
    output_best = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    top_left = max_loc
    bottom_right = (top_left[0] + scaled_w, top_left[1] + scaled_h)
    cv2.rectangle(output_best, top_left, bottom_right, (0, 0, 255), 3)

    axes[1, 2].imshow(cv2.cvtColor(output_best, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Best Match (red box)\nConfidence: {max_val:.3f}')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('heatmap_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved heatmap visualization to: heatmap_visualization.png")
    plt.show()


def show_multiscale_heatmap(image_path, template_path, scales=[0.5, 1.0, 1.5]):
    """Show heatmaps at multiple scales."""
    fig, axes = plt.subplots(len(scales), 3, figsize=(15, 5 * len(scales)))

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    if image is None or template is None:
        print("Error loading images!")
        return

    for idx, scale in enumerate(scales):
        # Resize template
        template_h, template_w = template.shape
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)

        if scaled_w > image.shape[1] or scaled_h > image.shape[0]:
            continue

        resized_template = cv2.resize(template, (scaled_w, scaled_h))

        # Perform template matching
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Template
        axes[idx, 0].imshow(resized_template, cmap='gray')
        axes[idx, 0].set_title(f'Template (scale={scale}x)')
        axes[idx, 0].axis('off')

        # Heatmap
        heatmap = axes[idx, 1].imshow(result, cmap='jet', interpolation='nearest')
        axes[idx, 1].set_title(f'Heatmap - Max: {max_val:.3f}')
        axes[idx, 1].axis('off')
        plt.colorbar(heatmap, ax=axes[idx, 1])

        # Detections
        threshold = 0.5
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            x, y = pt
            cv2.rectangle(output, pt, (pt[0] + scaled_w, pt[1] + scaled_h), (0, 255, 0), 2)

        axes[idx, 2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f'Detections: {len(locations[0])} matches')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('multiscale_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved multiscale heatmap visualization to: multiscale_heatmap.png")
    plt.show()


if __name__ == "__main__":
    import sys

    print("Heatmap Visualization for Sight Reader\n")
    print("Choose visualization:")
    print("  1. Single scale heatmap (scale=0.4)")
    print("  2. Multi-scale heatmap comparison")
    print()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-2): ")

    image_path = "examples/sheet_music.jpg"
    template_path = "templates/find.png"

    if choice == "1":
        show_heatmap_example(image_path, template_path, scale=0.4)
    elif choice == "2":
        show_multiscale_heatmap(image_path, template_path, scales=[0.3, 0.5, 0.7])
    else:
        print("Invalid choice. Showing single scale example...")
        show_heatmap_example(image_path, template_path, scale=0.4)
