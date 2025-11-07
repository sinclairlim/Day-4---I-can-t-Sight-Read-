import cv2
import numpy as np
import argparse
from pathlib import Path


class SightReaderV2:
    """
    Improved sight reader using staff line position analysis.
    Determines pitch by measuring notehead position relative to staff lines.
    """

    def __init__(self, template_dir="templates", confidence_threshold=0.5):
        """
        Initialize the sight reader.

        Args:
            template_dir: Directory containing find.png template
            confidence_threshold: Minimum confidence for notehead detection
        """
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self.find_template = None
        self.load_find_template()

    def load_find_template(self):
        """Load the find.png template for notehead detection."""
        find_path = self.template_dir / "find.png"
        if find_path.exists():
            self.find_template = cv2.imread(str(find_path), cv2.IMREAD_GRAYSCALE)
            print(f"Loaded notehead finder template")
        else:
            print(f"Warning: find.png not found in {self.template_dir}")

    def detect_noteheads(self, image):
        """
        Detect noteheads using find.png template matching.

        Args:
            image: Grayscale image of sheet music

        Returns:
            List of (x, y, radius) tuples for detected noteheads
        """
        if self.find_template is None:
            print("Warning: find.png template not loaded!")
            return []

        noteheads = []
        template_h, template_w = self.find_template.shape

        # Try multiple scales to handle different note sizes
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]

        all_detections = []

        for scale in scales:
            # Resize template
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)

            if scaled_w > image.shape[1] or scaled_h > image.shape[0]:
                continue

            resized_template = cv2.resize(self.find_template, (scaled_w, scaled_h))

            # Perform template matching
            result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

            # Find locations above threshold
            locations = np.where(result >= self.confidence_threshold)

            for pt in zip(*locations[::-1]):
                x, y = pt
                # Center of detection
                center_x = x + scaled_w // 2
                center_y = y + scaled_h // 2
                radius = max(scaled_w, scaled_h) // 2

                confidence = result[y, x]
                all_detections.append((center_x, center_y, radius, confidence))

        # Non-maximum suppression to remove overlapping detections
        noteheads = self.non_max_suppression_noteheads(all_detections)

        return noteheads

    def non_max_suppression_noteheads(self, detections, overlap_threshold=0.5):
        """Remove overlapping notehead detections."""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[3], reverse=True)

        keep = []

        while detections:
            best = detections.pop(0)
            keep.append((best[0], best[1], best[2]))  # x, y, r

            # Remove overlapping detections
            new_detections = []
            for det in detections:
                # Calculate distance between centers
                dist = np.sqrt((best[0] - det[0])**2 + (best[1] - det[1])**2)
                # If distance is greater than threshold, keep it
                if dist > (best[2] + det[2]) * overlap_threshold:
                    new_detections.append(det)

            detections = new_detections

        return keep

    def detect_staff_lines(self, image):
        """
        Detect horizontal staff lines and group them into staves.

        Args:
            image: Grayscale image of sheet music

        Returns:
            List of staves, where each staff is a list of 5 y-coordinates
        """
        # Apply binary threshold (invert so staff lines become white)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Row-wise density: fraction of pixels that belong to a horizontal line
        row_density = np.sum(binary == 255, axis=1).astype(np.float32) / binary.shape[1]

        # Smooth the density profile to suppress noise
        smoothed_density = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 9), 0).ravel()

        max_score = float(np.max(smoothed_density)) if smoothed_density.size else 0.0
        if max_score == 0:
            return []

        score_threshold = max_score * 0.4
        min_separation = max(4, int(image.shape[0] * 0.01))  # Avoid double-picking the same line

        # Detect local maxima above the threshold
        peaks = []
        for y in range(1, len(smoothed_density) - 1):
            score = smoothed_density[y]
            if score < score_threshold:
                continue
            if score >= smoothed_density[y - 1] and score >= smoothed_density[y + 1]:
                peaks.append((score, y))

        # Non-maximum suppression based on vertical spacing
        peaks.sort(reverse=True)  # Highest scores first
        staff_lines = []
        for score, y in peaks:
            if all(abs(y - existing) >= min_separation for existing in staff_lines):
                staff_lines.append(y)
        staff_lines.sort()

        staff_lines.sort()

        staves = []
        print(f"  Total staff lines detected: {len(staff_lines)}")
        print(f"  Staff lines: {staff_lines}")
        if len(staff_lines) < 5:
            return []

        spacings = np.diff(staff_lines)
        spacing_estimate = float(np.median(spacings)) if len(spacings) else 0.0
        if spacing_estimate == 0:
            spacing_estimate = 12.0
        max_staff_gap = spacing_estimate * 1.7

        # Slide through the detected lines and keep groups that look like real staves
        i = 0
        while i <= len(staff_lines) - 5:
            window = staff_lines[i:i + 5]
            window_spacings = np.diff(window)
            if np.max(window_spacings) <= max_staff_gap:
                staves.append(window)
                i += 5
            else:
                i += 1

        # Fallback: chunk sequentially if sliding detection failed
        if not staves:
            for i in range(0, len(staff_lines), 5):
                chunk = staff_lines[i:i + 5]
                if len(chunk) == 5:
                    staves.append(chunk)

        print(f"  Staves created: {len(staves)}")
        for i, staff in enumerate(staves):
            print(f"    Staff {i}: {staff}")

        return staves

    def find_staff_for_note(self, note_y, staves):
        """
        Find which staff a note belongs to based on its y-coordinate.
        Uses closest center distance to handle overlapping ranges.

        Args:
            note_y: Y-coordinate of the note
            staves: List of staves (each staff is 5 lines)

        Returns:
            Index of the staff, or None if not found
        """
        if not staves:
            return None

        # Find the closest staff by measuring distance to staff center
        min_distance = float('inf')
        closest_staff_idx = None

        for idx, staff in enumerate(staves):
            # Calculate staff center (middle line)
            staff_center = staff[2]  # Middle line (3rd of 5 lines)
            distance = abs(note_y - staff_center)

            if distance < min_distance:
                min_distance = distance
                closest_staff_idx = idx

        return closest_staff_idx

    def calculate_pitch_from_position(self, note_y, staff, clef="treble"):
        """
        Calculate pitch based on note position relative to staff lines using formula.

        Args:
            note_y: Y-coordinate of the note center
            staff: List of 5 staff line y-coordinates
            clef: "treble" or "bass"

        Returns:
            Pitch name (e.g., "C4", "E5")
        """
        if len(staff) != 5:
            return "?"

        # Calculate staff line spacing (should be around 14px)
        spacing = (staff[4] - staff[0]) / 4  # Distance between adjacent lines
        half_spacing = spacing / 2  # Half-space = 7px

        # Bottom line is staff[4] (5th line)
        bottom_line_y = staff[4]

        # Calculate position from bottom line
        # Position 0 = on bottom line, Position 1 = first space up, etc.
        position = (bottom_line_y - note_y) / half_spacing
        # Round to the nearest staff position to avoid biasing borderline detections
        position_rounded = round(position)

        # Use formula to calculate pitch for any position
        # Note sequence: C, D, E, F, G, A, B (repeating)
        note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

        if clef == "treble":
            # Treble clef: bottom line (staff[4]) = E4
            # Position 0 = E4, which is note index 2 (C=0, D=1, E=2)
            base_note_idx = 2  # E
            base_octave = 4
        else:  # bass clef
            # Bass clef: bottom line (staff[4]) = G2
            # Position 0 = G2, which is note index 4 (C=0, D=1, E=2, F=3, G=4)
            base_note_idx = 4  # G
            base_octave = 2

        # Calculate the absolute note index from position
        absolute_note_idx = base_note_idx + position_rounded

        # Calculate octave offset (every 7 notes = 1 octave)
        octave_offset = absolute_note_idx // 7
        note_idx = absolute_note_idx % 7

        # Calculate final octave
        octave = base_octave + octave_offset

        # Get note name
        note_name = note_names[note_idx]

        return f"{note_name}{octave}"

    def process_sheet_music(self, image_path, output_path=None, return_details=False):
        """
        Process a sheet music image and identify notes.

        Args:
            image_path: Path to the sheet music image
            output_path: Optional path to save the annotated image
            return_details: If True, also return note metadata for UI/reporting

        Returns:
            Annotated image with note labels
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect staff lines and group into staves
        print("Detecting staff lines...")
        staves = self.detect_staff_lines(gray)
        print(f"Found {len(staves)} staves (groups of 5 lines)")

        # Detect noteheads
        print("Detecting noteheads...")
        noteheads = self.detect_noteheads(gray)
        print(f"Found {len(noteheads)} noteheads")

        # Annotate image
        output_image = image.copy()

        note_details = []

        print("\nDetected notes:")
        print(f"{'X':<6} {'Y':<6} {'Staff':<7} {'Clef':<8} {'Pitch':<8}")
        print("-" * 45)

        for x, y, r in noteheads:
            note_clef = None
            note_position = None
            note_position_rounded = None

            # Draw circle around notehead
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)

            # Find which staff this note belongs to
            staff_idx = self.find_staff_for_note(y, staves)

            if staff_idx is not None and staff_idx < len(staves):
                staff = staves[staff_idx]

                # Determine clef based on staff position
                # Pattern: treble, bass, treble, bass (alternating)
                note_clef = "treble" if staff_idx % 2 == 0 else "bass"

                # Calculate pitch
                pitch = self.calculate_pitch_from_position(y, staff, note_clef)

                # Calculate position for debug
                bottom_line_y = staff[4]
                spacing = (staff[4] - staff[0]) / 4
                half_spacing = spacing / 2
                note_position = (bottom_line_y - y) / half_spacing
                note_position_rounded = round(note_position)

                # Print debug info
                print(f"{x:<6} {y:<6} {staff_idx:<7} {note_clef:<8} {pitch:<8} pos={note_position:.2f}→{note_position_rounded}")
            else:
                pitch = "?"
                print(f"{x:<6} {y:<6} {'None':<7} {'N/A':<8} {pitch:<8}")

            note_details.append({
                "x": x,
                "y": y,
                "radius": r,
                "staff": staff_idx,
                "clef": note_clef,
                "pitch": pitch,
                "position": note_position,
                "position_rounded": note_position_rounded,
            })

            # Draw label (above the notehead) - pitch only
            label = pitch
            cv2.putText(output_image, label, (x - r, y - r - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw staff lines for visualization
        for staff in staves:
            for line_y in staff:
                cv2.line(output_image, (0, line_y), (image.shape[1], line_y), (255, 0, 0), 1)

        # Save or display
        if output_path:
            cv2.imwrite(str(output_path), output_image)
            print(f"Saved annotated image to: {output_path}")

        if return_details:
            return output_image, note_details
        return output_image


def main():
    parser = argparse.ArgumentParser(description="Sight Reading Helper V2 - Position-based pitch detection")
    parser.add_argument("--input", "-i", required=True, help="Path to sheet music image")
    parser.add_argument("--output", "-o", help="Path to save annotated image")
    parser.add_argument("--templates", "-t", default="templates", help="Template directory")

    args = parser.parse_args()

    # Create sight reader
    reader = SightReaderV2(template_dir=args.templates)

    # Process image
    output_image = reader.process_sheet_music(args.input, args.output)

    # Display result
    print("Displaying result... Close the window to exit.")
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Sight Reading Helper V2 - Detected Notes')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
