"""
Visualize heatmap of a specific pitch template (e.g., F1) against sheet music.
Shows where the template matches across the entire score.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def show_pitch_heatmap(sheet_music_path, template_path, template_name):
    """
    Show heatmap of template matching for a specific pitch.

    Args:
        sheet_music_path: Path to sheet music image
        template_path: Path to pitch template (e.g., F1.png)
        template_name: Name of the template for display
    """
    # Load images
    sheet = cv2.imread(str(sheet_music_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    if sheet is None or template is None:
        print(f"Error loading images!")
        print(f"Sheet music: {sheet_music_path} - {'OK' if sheet is not None else 'FAILED'}")
        print(f"Template: {template_path} - {'OK' if template is not None else 'FAILED'}")
        return

    print(f"Sheet music size: {sheet.shape}")
    print(f"Template size: {template.shape}")

    # Perform template matching
    result = cv2.matchTemplate(sheet, template, cv2.TM_CCOEFF_NORMED)

    # Find min/max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Match scores - Min: {min_val:.3f}, Max: {max_val:.3f}")

    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Original sheet music
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(sheet, cmap='gray')
    ax1.set_title('Original Sheet Music', fontsize=14)
    ax1.axis('off')

    # 2. Template
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(template, cmap='gray')
    ax2.set_title(f'Template: {template_name}\nSize: {template.shape[1]}x{template.shape[0]}', fontsize=12)
    ax2.axis('off')

    # 3. Full heatmap
    ax3 = fig.add_subplot(gs[1, 1:])
    heatmap = ax3.imshow(result, cmap='jet', interpolation='nearest', aspect='auto')
    ax3.set_title(f'Similarity Heatmap (Full)\nMax: {max_val:.3f}, Min: {min_val:.3f}', fontsize=12)
    ax3.axis('off')
    plt.colorbar(heatmap, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Thresholded heatmap
    threshold = 0.6
    ax4 = fig.add_subplot(gs[2, 0])
    result_thresh = result.copy()
    result_thresh[result_thresh < threshold] = 0
    heatmap_thresh = ax4.imshow(result_thresh, cmap='hot', interpolation='nearest', aspect='auto')
    ax4.set_title(f'Heatmap (threshold >= {threshold})', fontsize=12)
    ax4.axis('off')
    plt.colorbar(heatmap_thresh, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Detections overlaid on sheet music
    ax5 = fig.add_subplot(gs[2, 1:])
    output = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)

    # Find all matches above threshold
    locations = np.where(result >= threshold)
    match_count = len(locations[0])

    template_h, template_w = template.shape

    for pt in zip(*locations[::-1]):
        x, y = pt
        cv2.rectangle(output, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 0), 2)

    ax5.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax5.set_title(f'Detections (threshold >= {threshold})\nFound: {match_count} matches', fontsize=12)
    ax5.axis('off')

    plt.suptitle(f'Template Matching Analysis: {template_name}', fontsize=16, y=0.98)
    plt.tight_layout()

    output_file = f'pitch_heatmap_{template_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.show()


def show_multiple_pitch_heatmaps(sheet_music_path, template_dir, pitches):
    """Show heatmaps for multiple pitches side by side."""
    sheet = cv2.imread(str(sheet_music_path), cv2.IMREAD_GRAYSCALE)

    if sheet is None:
        print(f"Error loading sheet music: {sheet_music_path}")
        return

    fig, axes = plt.subplots(len(pitches), 2, figsize=(18, 6 * len(pitches)))

    if len(pitches) == 1:
        axes = [axes]

    for idx, pitch in enumerate(pitches):
        template_path = Path(template_dir) / f"{pitch}.png"
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

        if template is None:
            print(f"Could not load template: {template_path}")
            continue

        # Perform template matching
        result = cv2.matchTemplate(sheet, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Heatmap
        heatmap = axes[idx, 0].imshow(result, cmap='jet', interpolation='nearest', aspect='auto')
        axes[idx, 0].set_title(f'{pitch} - Heatmap (Max: {max_val:.3f})')
        axes[idx, 0].axis('off')
        plt.colorbar(heatmap, ax=axes[idx, 0], fraction=0.046, pad=0.04)

        # Detections
        threshold = 0.6
        output = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        locations = np.where(result >= threshold)
        template_h, template_w = template.shape

        for pt in zip(*locations[::-1]):
            x, y = pt
            cv2.rectangle(output, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 0), 2)

        axes[idx, 1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'{pitch} - Detections ({len(locations[0])} matches)')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig('multiple_pitch_heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: multiple_pitch_heatmaps.png")
    plt.show()


if __name__ == "__main__":
    sheet_music = "examples/sheet_music.jpg"
    template_dir = "templates"

    if len(sys.argv) > 1:
        # Single pitch mode
        pitch = sys.argv[1]
        template_path = f"{template_dir}/{pitch}.png"
        print(f"Analyzing template matching for: {pitch}")
        show_pitch_heatmap(sheet_music, template_path, pitch)
    else:
        # Default: show F1
        pitch = "F1"
        template_path = f"{template_dir}/{pitch}.png"
        print(f"Analyzing template matching for: {pitch}")
        print("Usage: python show_pitch_heatmap.py [pitch_name]")
        print("Example: python show_pitch_heatmap.py F1")
        print()
        show_pitch_heatmap(sheet_music, template_path, pitch)
