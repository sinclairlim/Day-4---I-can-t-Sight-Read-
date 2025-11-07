"""
Helper script to create template images from a sample sheet music.
You can use this to extract note images that will serve as templates.
"""

import cv2
import numpy as np
from pathlib import Path


def extract_template_interactive(image_path, output_dir="templates"):
    """
    Interactive tool to extract note templates from sheet music.

    Instructions:
    1. Click and drag to select a region containing a note
    2. Press 's' to save the selected region
    3. Press 'q' to quit
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clone = image.copy()

    # Selection state
    selecting = False
    start_point = None
    end_point = None
    template_count = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_point, end_point, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_point = (x, y)
            end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_point = (x, y)
            clone = image.copy()
            cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_point = (x, y)
            clone = image.copy()
            cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)

    cv2.namedWindow("Extract Template")
    cv2.setMouseCallback("Extract Template", mouse_callback)

    print("\n=== Template Extraction Tool ===")
    print("Instructions:")
    print("  - Click and drag to select a note")
    print("  - Press 's' to save the selection")
    print("  - Press 'r' to reset selection")
    print("  - Press 'q' to quit")
    print("================================\n")

    while True:
        cv2.imshow("Extract Template", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and start_point and end_point:
            # Extract and save template
            x1, y1 = start_point
            x2, y2 = end_point

            # Ensure correct ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum size check
                template = gray[y1:y2, x1:x2]

                # Ask for template name
                template_name = input(f"Enter name for this template (e.g., quarter_note): ")
                if template_name:
                    filename = output_path / f"{template_name}.png"
                    cv2.imwrite(str(filename), template)
                    print(f"Saved template: {filename}")
                    template_count += 1

                    # Reset selection
                    start_point = None
                    end_point = None
                    clone = image.copy()
            else:
                print("Selection too small, try again")

        elif key == ord('r'):
            # Reset selection
            start_point = None
            end_point = None
            clone = image.copy()

    cv2.destroyAllWindows()
    print(f"\nExtracted {template_count} templates")


def create_sample_templates():
    """
    Create basic synthetic templates for common note types.
    This is a fallback if you don't have sample sheet music to extract from.
    """
    output_dir = Path("templates")
    output_dir.mkdir(exist_ok=True)

    print("Creating synthetic note templates...")

    # Quarter note (filled notehead with stem)
    quarter = np.ones((40, 25), dtype=np.uint8) * 255
    cv2.ellipse(quarter, (12, 30), (8, 6), 0, 0, 360, 0, -1)  # Filled head
    cv2.rectangle(quarter, (20, 5), (22, 30), 0, -1)  # Stem
    cv2.imwrite(str(output_dir / "quarter_note.png"), quarter)

    # Half note (hollow notehead with stem)
    half = np.ones((40, 25), dtype=np.uint8) * 255
    cv2.ellipse(half, (12, 30), (8, 6), 0, 0, 360, 0, 2)  # Hollow head
    cv2.rectangle(half, (20, 5), (22, 30), 0, -1)  # Stem
    cv2.imwrite(str(output_dir / "half_note.png"), half)

    # Whole note (hollow notehead, no stem)
    whole = np.ones((20, 20), dtype=np.uint8) * 255
    cv2.ellipse(whole, (10, 10), (9, 6), 0, 0, 360, 0, 2)  # Hollow head
    cv2.imwrite(str(output_dir / "whole_note.png"), whole)

    print(f"Created 3 synthetic templates in {output_dir}/")
    print("Note: These are basic templates. For best results, extract templates from actual sheet music.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Extract from provided image
        extract_template_interactive(sys.argv[1])
    else:
        # Create synthetic templates
        print("No image provided. Creating synthetic templates...")
        print("To extract from your own sheet music, run:")
        print("  python create_templates.py your_sheet_music.jpg")
        print()
        create_sample_templates()
