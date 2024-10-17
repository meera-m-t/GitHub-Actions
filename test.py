import numpy as np
from collections import Counter

class FinProcess:
    def __init__(
        self,
        fins_detector_model_path: str,
        interpolation_points: int,
        pixel_thresh: int,
    ):
        """
        Initialize the FinProcess class for handling fin detection and processing.

        Parameters:
        fins_detector_model_path (str): Path to the fin detection model.
        interpolation_points (int): Number of additional interpolation points between polygon vertices.
        pixel_thresh (int): Pixel threshold for fin point hit detection.
        """
        self.model = self._load_model(fins_detector_model_path)
        self.interpolation_points = interpolation_points
        self.pixel_thresh = pixel_thresh
        self.fin_coordinates: Optional[np.ndarray] = None
        self.previous_fin_coordinates: Optional[np.ndarray] = None
        self.total_points = 0
        self.covered_points: set[tuple[float, float]] = set()

        # Store history of detected classes
        self.class_history = []

    def _load_model(self, model_path: str) -> YOLO:
        """Load the fin detection model."""
        return YOLO(model_path)

    def _interpolate_polygon_points(
        self, points: np.ndarray, additional_points: int
    ) -> np.ndarray:
        """Interpolate points between the vertices of the fin."""
        new_points = []
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]  # Wrap around to the first point
            new_points.append(start_point)

            # Interpolate between points
            for j in range(1, additional_points + 1):
                fraction = j / (additional_points + 1)
                x = start_point[0] + (end_point[0] - start_point[0]) * fraction
                y = start_point[1] + (end_point[1] - start_point[1]) * fraction
                new_points.append([x, y])

        return np.array(new_points)

    def _process_detections(self, detections):
        """
        Process detections to separate fin and surface.

        detections: Object detected by the model.
        Returns:
        - fin_index: Index of the detected fin.
        - surface_index: Index of the detected surface.
        - majority_class: The most frequent fin class detected (0, 1, 2 for large, medium, small)
        """
        fin_index = None
        surface_index = None
        detected_classes = []

        # Assuming detections have a class attribute or labels
        classes = detections[0].obb.cls.cpu().numpy()  # Adjust this to fit your model's detection structure
        for i, cls in enumerate(classes):
            if cls in [0, 1, 2]:  # Fin classes (0 = large, 1 = medium, 2 = small)
                fin_index = i
                detected_classes.append(cls)  # Track the detected fin class
            elif cls == 3:  # Surface class
                surface_index = i

        # Update class history and keep track of majority class
        self.class_history.extend(detected_classes)

        if self.class_history:
            # Calculate the majority class
            class_counts = Counter(self.class_history)
            majority_class = class_counts.most_common(1)[0][0]  # Get the most common class
        else:
            majority_class = None

        return fin_index, surface_index, majority_class



import numpy as np
from ultralytics import YOLO

from src.mahle_final_inspection.sub_process.distance_calculation import DistanceCalculation
from src.mahle_final_inspection.sub_process.fin_process import FinProcess
from src.mahle_final_inspection.sub_process.orientation_check import OrientationCheck

# Constants for areas based on fin classes
AREA_LARGE = 1000  # Example values, replace with actual ones
AREA_MEDIUM = 500
AREA_SMALL = 200

class HelicoilDepthCheck:
    def __init__(
        self,
        fins_detector_model_path: str,
        driver_detector_model_path: str,
        interpolation_points: int,
        pixel_thresh: int,
        coverage_perc: float,
        imgsz_fin: int,
        imgsz_driver: int,
        conf_fin: float,
        conf_driver: float,
        verbose: bool,
    ):
        """Initialize the Helicoil depth check process."""
        self.fin_process = FinProcess(fins_detector_model_path, interpolation_points, pixel_thresh)
        self.driver_model = self._load_model(driver_detector_model_path)
        self.orientation_check = OrientationCheck()
        self.distance_calculation = DistanceCalculation()

        self.pixel_thresh = pixel_thresh
        self.coverage_perc = coverage_perc
        self.last_fin_coords = None
        self.last_fin_area = None  # To store last detected fin area
        self.global_min_distances: list[float] = []
        self.global_per_point_min: list[np.ndarray] = []
        self.both_detected = False
        self.imgsz_fin = imgsz_fin
        self.imgsz_driver = imgsz_driver
        self.conf_fin = conf_fin
        self.conf_driver = conf_driver
        self.verbose = verbose

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLO model."""
        return YOLO(model_path)

    def _find_driver(self, frame: np.ndarray, imgsz: int, conf: float, verbose: bool) -> np.ndarray:
        """Find the driver using the YOLO model."""
        detections = self.driver_model(frame, imgsz=imgsz, conf=conf, verbose=verbose)
        if len(detections) > 0 and hasattr(detections[0], "obb"):
            classes = detections[0].obb.cls.cpu().numpy()
            for c in classes:
                if c == 1:
                    obb = detections[0].obb.xyxyxyxy.cpu().numpy()[0]
                    return obb.reshape(-1, 2)
        return None

    def _calculate_area(self, coordinates: np.ndarray) -> float:
        """Calculate the area of the fin based on its coordinates."""
        if coordinates is None:
            return 0.0
        # Assuming coordinates form a closed polygon, calculate the area
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _get_constant_area(self, cls: int) -> float:
        """Return the area constant based on the fin class."""
        if cls == 0:  # Large fin
            return AREA_LARGE
        elif cls == 1:  # Medium fin
            return AREA_MEDIUM
        elif cls == 2:  # Small fin
            return AREA_SMALL
        else:
            return 0.0  # Return 0 if class doesn't match

    def _check_operator(self, frame: np.ndarray, frame_num: int):
        """Determine if the driver is near the fin and track the interaction."""
        detections = self.fin_process.model(frame, imgsz=self.imgsz_fin, conf=self.conf_fin, verbose=self.verbose)
        fin_index, surface_index, classes = self.fin_process._process_detections(detections)
        fin_coords = self.fin_process.process_fin(frame, detections, fin_index)
        driver_coords = self._find_driver(frame, imgsz=self.imgsz_driver, conf=self.conf_driver, verbose=self.verbose)

        if fin_coords is not None and driver_coords is not None and not self.both_detected:
            print("Both fin and driver detected for the first time!")
            self.both_detected = True

        if self.both_detected and fin_coords is not None:
            new_area = self._calculate_area(fin_coords)
            constant_area = self._get_constant_area(classes[fin_index])

            if self.orientation_check._check_flip(fin_coords):
                # Flip detected: directly update the coordinates without comparison
                print("Flip detected. Updating coordinates directly.")
                self.last_fin_coords = fin_coords
                self.last_fin_area = new_area
            else:
                # No flip: check if the new area is closer to the constant area
                if self.last_fin_area is None or abs(new_area - constant_area) < abs(self.last_fin_area - constant_area):
                    # New area is closer to the constant area, update coordinates
                    print("New area is closer to the constant area. Updating coordinates.")
                    self.last_fin_coords = fin_coords
                    self.last_fin_area = new_area
                else:
                    # Keep the previous coordinates
                    print("Keeping previous coordinates.")
                    fin_coords = self.last_fin_coords

            driver_coords = self._find_driver(frame, imgsz=self.imgsz_driver, conf=self.conf_driver, verbose=self.verbose)

            if fin_coords is not None and driver_coords is not None:
                distances = np.sqrt(np.sum((np.array(fin_coords)[:, None, :] - np.array(driver_coords)[None, :, :]) ** 2, axis=2))

                if distances.size > 0:
                    min_distances = np.min(distances)
                    self.global_min_distances.append(min_distances)
                    per_point_min_distances = np.min(distances, axis=1)
                    self.global_per_point_min.append(per_point_min_distances)

                    pixels = np.where(per_point_min_distances <= self.fin_process.pixel_thresh)[0]
                    self.fin_process.covered_points.update(pixels)

    def final_decision(self) -> bool:
        """Return the final decision based on total coverage and print global minimums."""
        if self.fin_process.total_points > 0:
            coverage = len(self.fin_process.covered_points) / self.fin_process.total_points
            return coverage >= self.coverage_perc
        return False

    def close(self):
        """Release resources."""
        self.video_writer.release()

    def inspect_depth(self, frame: np.ndarray, frame_num: int):
        """Analyze each frame where the driver and hand are detected."""
        self._check_operator(frame, frame_num)












