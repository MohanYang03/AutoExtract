from __future__ import annotations

import os
import pathlib

import cv2
import torch
import pytesseract
from typing import List, Dict, Any

from vision_utils import crop_with_padding, ocr_text, ocr_number


class BarGraphAnalyzer:
    """
    Analyze bar graphs using a YOLO model plus OCR to extract structure and values.

    Responsibilities:
    - Load a YOLO model for object detection on bar graphs
    - Detect key components: bars, ymax, origin, axes, labels, x-groups
    - Extract numeric values and text via OCR
    - Provide structured results and utilities for downstream analysis
    - Track and update box coordinates when modified
    """

    def __init__(self, model_path: str, class_names: List[str], pytesseract_cmd: str | None = None) -> None:
        """
        Initialize the analyzer with model and class metadata.

        Args:
            model_path: Filesystem path to YOLO weights (.pt).
            class_names: Mapping of class indices to human-readable labels.
            pytesseract_cmd: Optional path to the tesseract binary.
        """
        if pytesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_cmd

        # Load YOLOv5 custom model from torch hub (repo is cached after first download; Docker image pre-warms this).
        # best.pt was pickled on macOS/Linux and contains PosixPath objects, which can't be
        # instantiated on Windows; map them to WindowsPath for the duration of the load.
        posix_path = pathlib.PosixPath
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            self.model: Any = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=model_path,
                trust_repo=True,
            )
        finally:
            pathlib.PosixPath = posix_path
        self.class_names = class_names

        # Runtime state populated per image
        self.image: Any | None = None
        self.detections: List[Dict[str, Any]] | None = None

        # Semantic elements
        self.origin_value: float | str = 0
        self.ymax_value: float | str = 0
        self.origin_conversion_error: str | None = None
        self.ymax_conversion_error: str | None = None
        self.x_label_texts: List[str] = []
        self.label_text: str = ""

        # All detection types as dictionaries with unique IDs
        self.bars: Dict[str, Dict[str, Any]] = {}
        self.uptails: Dict[str, Dict[str, Any]] = {}
        self.yaxis: Dict[str, Dict[str, Any]] = {}
        self.xaxis: Dict[str, Dict[str, Any]] = {}
        self.origins: Dict[str, Dict[str, Any]] = {}
        self.ymaxes: Dict[str, Dict[str, Any]] = {}
        self.labels: Dict[str, Dict[str, Any]] = {}
        self.x_groups: Dict[str, Dict[str, Any]] = {}
        self.legends: Dict[str, Dict[str, Any]] = {}
        self.legend_groups: Dict[str, Dict[str, Any]] = {}
        
    @property
    def _category_dicts(self) -> List[Dict[str, Any]]:
        """
        Dynamically return the current category dictionaries.
        This ensures we always have the latest references.
        """
        return [
            self.bars, self.uptails, self.yaxis, self.xaxis,
            self.origins, self.ymaxes, self.labels,
            self.x_groups, self.legends, self.legend_groups
        ]

    def detect_box(self, image_path: str) -> None:
        """
        Run detection on an input image and populate analyzer fields.

        Args:
            image_path: Path to the image file to analyze
        """
        # Load the image
        self.image = cv2.imread(image_path)

        # Run YOLO model on the image
        results = self.model(self.image)

        # Extract the detections (x1, y1, x2, y2, conf, cls)
        detections = results.xyxy[0].numpy()
        self.detections = [
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf),
                "class": int(cls),
                "label": self.class_names[int(cls)],
            }
            for x1, y1, x2, y2, conf, cls in detections
        ]

        # First organize detections with IDs
        self._organize_detections()

        # Extract single objects for calculations (since we need one of each type)
        yaxis = next(iter(self.yaxis.values())) if self.yaxis else None
        xaxis = next(iter(self.xaxis.values())) if self.xaxis else None
        origin = next(iter(self.origins.values())) if self.origins else None
        ymax = next(iter(self.ymaxes.values())) if self.ymaxes else None
        label = next(iter(self.labels.values())) if self.labels else None
        x_groups = list(self.x_groups.values())

        # Component validation now handled separately via get_component_status()
        # No validation errors raised here - let frontend handle component checking

        # OCR numeric values and validate conversion (only if components exist)
        origin_text = self.extract_numbers(self.image, origin) if origin else ''
        ymax_text = self.extract_numbers(self.image, ymax) if ymax else ''
        
        # Try to convert to numbers and store conversion status
        self.origin_value = origin_text
        self.ymax_value = ymax_text
        self.origin_conversion_error = None
        self.ymax_conversion_error = None
        
        try:
            float(origin_text) if origin_text else None
        except ValueError:
            self.origin_conversion_error = f"Manual input needed for origin value: '{origin_text}'"
            
        try:
            float(ymax_text) if ymax_text else None
        except ValueError:
            self.ymax_conversion_error = f"Manual input needed for ymax value: '{ymax_text}'"
        
        # OCR x-axis labels and main label (only if components exist)
        self.x_label_texts = []
        if x_groups:
            for x_label in x_groups:
                self.x_label_texts.append(self.extract_text(self.image, x_label))

        if label is not None:
            self.label_text = self.extract_text(self.image, label, rotate=True)

    def get_component_status(self) -> Dict[str, Any]:
        """
        Get the status of all required components for height calculation.
        Considers both physical detection and manual input values.
        
        Returns:
            Dictionary with component status and missing components list.
        """
        required_components = {
            'yaxis': self.yaxis,
            'xaxis': self.xaxis,
            'bars': self.bars,
            'ymax': self.ymaxes,
            'origin': self.origins
        }
        
        # Check which components are present
        present_components = {}
        missing_components = []
        
        for component_name, component_dict in required_components.items():
            if component_dict and len(component_dict) > 0:
                present_components[component_name] = len(component_dict)
            else:
                missing_components.append(component_name)
        
        # Special handling for ymax and origin: consider manual input values
        if 'ymax' in missing_components and self.ymax_value and self.ymax_value.strip():
            try:
                float(self.ymax_value)
                missing_components.remove('ymax')
                present_components['ymax'] = 'manual_input'
            except ValueError:
                pass  # Keep as missing if value can't be converted to float
                
        if 'origin' in missing_components and self.origin_value and self.origin_value.strip():
            try:
                float(self.origin_value)
                missing_components.remove('origin')
                present_components['origin'] = 'manual_input'
            except ValueError:
                pass  # Keep as missing if value can't be converted to float
        
        return {
            'present_components': present_components,
            'missing_components': missing_components,
            'all_components_ready': len(missing_components) == 0
        }

    def _organize_detections(self) -> None:
        """
        Organize detections into dictionaries with unique IDs and sort them.
        This method creates unique identifiers for each detection and organizes
        them into the appropriate category dictionaries.
        """
        # Clear previous organization
        self.bars = {}
        self.uptails = {}
        self.yaxis = {}
        self.xaxis = {}
        self.origins = {}
        self.ymaxes = {}
        self.labels = {}
        self.x_groups = {}
        self.legends = {}
        self.legend_groups = {}
        
        # Organize detections by label using a mapping
        detection_lists = {
            'bar': [],
            'uptail': [],
            'yaxis': [],
            'xaxis': [],
            'origin': [],
            'ymax': [],
            'label': [],
            'x_group': [],
            'legend': [],
            'legend_group': []
        }
        
        for i, detection in enumerate(self.detections):
            label = detection['label']
            if label in detection_lists:
                detection_lists[label].append(detection)
                # Store original index for stable identification
                detection['original_index'] = i
        
        # Sort by x1 coordinate where applicable
        sorted_bars = sorted(detection_lists['bar'], key=lambda bar: bar['x1'])
        sorted_uptails = sorted(detection_lists['uptail'], key=lambda uptail: uptail['x1'])
        sorted_x_groups = sorted(detection_lists['x_group'], key=lambda x_group: x_group['x1'])
        
        # Assign sequential IDs and organize
        for i, bar in enumerate(sorted_bars):
            new_id = f"bar_{i+1}"
            bar['id'] = new_id
            self.bars[new_id] = bar
            
        for i, uptail in enumerate(sorted_uptails):
            new_id = f"uptail_{i+1}"
            uptail['id'] = new_id
            self.uptails[new_id] = uptail
            
        for i, yaxis in enumerate(detection_lists['yaxis']):
            new_id = f"yaxis_{i+1}"
            yaxis['id'] = new_id
            self.yaxis[new_id] = yaxis
            
        for i, xaxis in enumerate(detection_lists['xaxis']):
            new_id = f"xaxis_{i+1}"
            xaxis['id'] = new_id
            self.xaxis[new_id] = xaxis
            
        for i, origin in enumerate(detection_lists['origin']):
            new_id = f"origin_{i+1}"
            origin['id'] = new_id
            self.origins[new_id] = origin
            
        for i, ymax in enumerate(detection_lists['ymax']):
            new_id = f"ymax_{i+1}"
            ymax['id'] = new_id
            self.ymaxes[new_id] = ymax
            
        for i, label in enumerate(detection_lists['label']):
            new_id = f"label_{i+1}"
            label['id'] = new_id
            self.labels[new_id] = label
            
        for i, x_group in enumerate(sorted_x_groups):
            new_id = f"x_group_{i+1}"
            x_group['id'] = new_id
            self.x_groups[new_id] = x_group
            
        for i, legend in enumerate(detection_lists['legend']):
            new_id = f"legend_{i+1}"
            legend['id'] = new_id
            self.legends[new_id] = legend
            
        for i, legend_group in enumerate(detection_lists['legend_group']):
            new_id = f"legend_group_{i+1}"
            legend_group['id'] = new_id
            self.legend_groups[new_id] = legend_group

    def update_box_coordinates(self, box_id: str, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        Update the coordinates of a specific detection box.
        
        Args:
            box_id: Unique identifier of the box to update
            x1: New left coordinate
            y1: New top coordinate  
            x2: New right coordinate
            y2: New bottom coordinate
            
        Returns:
            bool: True if update was successful, False if box not found
            
        Raises:
            ValueError: If coordinates are invalid
        """
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid coordinates: x1 must be < x2 and y1 must be < y2")
            
        # Find which category the box belongs to and update it
        new_coords = {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
        
        # Check all category dictionaries
        for category_dict in self._category_dicts:
            if box_id in category_dict:
                category_dict[box_id].update(new_coords)
                return True
        
        return False

    def get_box_by_id(self, box_id: str) -> Dict[str, Any] | None:
        """
        Get a detection box by its unique ID.
        
        Args:
            box_id: Unique identifier of the box
            
        Returns:
            Dict[str, Any] | None: Box data or None if not found
        """
        # Search in all category dictionaries
        for category_dict in self._category_dicts:
            if box_id in category_dict:
                return category_dict[box_id]
        
        return None

    def get_all_boxes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all detection boxes organized by category.
        
        Returns:
            Dict containing all detection categories with their boxes
        """
        # Create a mapping of category names to their dictionaries
        category_names = ['bars', 'uptails', 'yaxis', 'xaxis', 'origins', 
                         'ymaxes', 'labels', 'x_groups', 'legends', 'legend_groups']
        return dict(zip(category_names, self._category_dicts))





    def extract_text(self, image, bbox, rotate: bool = False) -> str:
        """
        Extract general text from a region defined by bbox.

        Args:
            image: Full BGR image
            bbox: Detection bbox dict with x1,y1,x2,y2
            rotate: Whether to rotate when text is vertical
        """
        region = crop_with_padding(image, bbox)
        return ocr_text(region, rotate_clockwise=rotate)

    def extract_numbers(self, image, bbox, rotate: bool = False) -> str:
        """
        Extract numeric text from a region defined by bbox.

        Args:
            image: Full BGR image
            bbox: Detection bbox dict with x1,y1,x2,y2
            rotate: Whether to rotate when number runs vertically
        """
        region = crop_with_padding(image, bbox)
        return ocr_number(region, rotate_clockwise=rotate)

    def calculate_heights(self) -> Dict[str, Any]:
        """
        Calculate bar and uptail heights using detected geometry and scale.

        Returns:
            Dictionary keyed by chart label with computed arrays and metadata.
        """
        results: Dict[str, Any] = {}

        # Convert to numbers (validation handled on frontend)
        origin_value = float(self.origin_value)
        ymax_value = float(self.ymax_value)

        # y-axis height in pixels (top - bottom). Get the first yaxis detection
        if not self.yaxis:
            raise ValueError("yaxis not set")
        yaxis_obj = next(iter(self.yaxis.values()))
        yaxis_height = yaxis_obj['y1'] - yaxis_obj['y2']
        scale_factor = (ymax_value - origin_value) / yaxis_height

        bar_heights: List[float] = []
        for bar in self.bars.values():
            bar_ymax = bar['y1']
            bar_ymin = bar['y2']
            height = (bar_ymax - bar_ymin) * scale_factor + origin_value
            bar_heights.append(height)

        uptail_heights: List[float] = []
        for uptail in self.uptails.values():
            uptail_ymax = uptail['y1']
            uptail_ymin = uptail['y2']
            height = (uptail_ymax - uptail_ymin) * scale_factor
            uptail_heights.append(height)

        # Generate bar names - use x_label_texts if available, otherwise generate default names
        bar_names = []
        num_bars = len(bar_heights)
        
        if self.x_label_texts and len(self.x_label_texts) >= num_bars:
            # Use existing x_label_texts
            bar_names = self.x_label_texts[:num_bars]
        else:
            # Generate default names: Bar 1, Bar 2, etc.
            bar_names = [f"Bar {i+1}" for i in range(num_bars)]
            # Update self.x_label_texts with the generated names
            self.x_label_texts = bar_names.copy()

        results[self.label_text] = {
            "bar_heights": bar_heights,
            "uptail_heights": uptail_heights,
            "bar_names": bar_names,  # Use the generated/updated names
        }
        return results

    def update_bar_names(self, bar_names: List[str]) -> bool:
        """
        Update the bar names in x_label_texts.
        
        Args:
            bar_names: List of new bar names
            
        Returns:
            bool: True if update was successful
        """
        try:
            if len(bar_names) != len(self.bars):
                return False
            
            self.x_label_texts = bar_names.copy()
            return True
        except Exception:
            return False

    def change_box_category(self, box_id: str, new_category: str) -> bool:
        """
        Change the category of a detection box and reorganize the data structures.
        
        Args:
            box_id: Unique identifier of the box to change
            new_category: New category for the box
            
        Returns:
            bool: True if the category change was successful
        """
        try:
            # Find the box by original index in the detections list
            box_data = None
            old_category = None
            
            # Extract the index from the box_id (e.g., "label_1" -> 1, "ymax_2" -> 2)
            try:
                # Try to find by the generated ID first (for backward compatibility)
                for category_dict in self._category_dicts:
                    if box_id in category_dict:
                        box_data = category_dict[box_id]
                        # Determine which category this was
                        if category_dict is self.bars:
                            old_category = 'bar'
                        elif category_dict is self.uptails:
                            old_category = 'uptail'
                        elif category_dict is self.yaxis:
                            old_category = 'yaxis'
                        elif category_dict is self.xaxis:
                            old_category = 'xaxis'
                        elif category_dict is self.origins:
                            old_category = 'origin'
                        elif category_dict is self.ymaxes:
                            old_category = 'ymax'
                        elif category_dict is self.labels:
                            old_category = 'label'
                        elif category_dict is self.x_groups:
                            old_category = 'x_group'
                        elif category_dict is self.legends:
                            old_category = 'legend'
                        elif category_dict is self.legend_groups:
                            old_category = 'legend_group'
                        break
                
                # If not found by ID, try to find by original index
                if not box_data:
                    # Parse the ID to get the index (e.g., "label_1" -> 1)
                    if '_' in box_id:
                        category_name, index_str = box_id.rsplit('_', 1)
                        try:
                            index = int(index_str) - 1  # Convert to 0-based index
                            # Find the box in the original detections list
                            for detection in self.detections:
                                if detection.get('original_index') == index:
                                    box_data = detection
                                    old_category = detection.get('label', 'unknown')
                                    break
                        except ValueError:
                            pass
            except Exception as e:
                print(f"Error parsing box_id {box_id}: {e}")
                return False
            
            if not box_data:
                print(f"Box with ID {box_id} not found in any category")
                print(f"Available categories and counts:")
                print(f"  bars: {len(self.bars)}")
                print(f"  uptails: {len(self.uptails)}")
                print(f"  yaxis: {len(self.yaxis)}")
                print(f"  xaxis: {len(self.xaxis)}")
                print(f"  origins: {len(self.origins)}")
                print(f"  ymaxes: {len(self.ymaxes)}")
                print(f"  labels: {len(self.labels)}")
                print(f"  x_groups: {len(self.x_groups)}")
                print(f"  legends: {len(self.legends)}")
                print(f"  legend_groups: {len(self.legend_groups)}")
                return False
            
            # Remove from old category
            if old_category == 'bar':
                del self.bars[box_id]
            elif old_category == 'uptail':
                del self.uptails[box_id]
            elif old_category == 'yaxis':
                del self.yaxis[box_id]
            elif old_category == 'xaxis':
                del self.xaxis[box_id]
            elif old_category == 'origin':
                del self.origins[box_id]
            elif old_category == 'ymax':
                del self.ymaxes[box_id]
            elif old_category == 'label':
                del self.labels[box_id]
            elif old_category == 'x_group':
                del self.x_groups[box_id]
            elif old_category == 'legend':
                del self.legends[box_id]
            elif old_category == 'legend_group':
                del self.legend_groups[box_id]
            
            # Update the box's label
            box_data['label'] = new_category
            
            # Add to new category but keep the original ID for consistency
            if new_category == 'bar':
                self.bars[box_id] = box_data
            elif new_category == 'uptail':
                self.uptails[box_id] = box_data
            elif new_category == 'yaxis':
                self.yaxis[box_id] = box_data
            elif new_category == 'xaxis':
                self.xaxis[box_id] = box_data
            elif new_category == 'origin':
                self.origins[box_id] = box_data
            elif new_category == 'ymax':
                self.ymaxes[box_id] = box_data
            elif new_category == 'label':
                self.labels[box_id] = box_data
            elif new_category == 'x_group':
                self.x_groups[box_id] = box_data
            elif new_category == 'legend':
                self.legends[box_id] = box_data
            elif new_category == 'legend_group':
                self.legend_groups[box_id] = box_data
            
            return True
            
        except Exception as e:
            print(f"Error changing box category: {e}")
            return False

    def _next_index(self, prefix: str) -> int:
        """
        Next unused number for an ID like 'bar_N'. Counting the dict (len + 1)
        reused live IDs after a deletion and silently overwrote a box; boxes
        also keep their old ID when their category changes, so check every dict.
        """
        used = [0]
        for category_dict in self._category_dicts:
            for box_id in category_dict:
                head, _, tail = box_id.rpartition('_')
                if head == prefix and tail.isdigit():
                    used.append(int(tail))
        return max(used) + 1

    def add_new_box(self, box_data: Dict[str, Any]) -> str:
        """
        Add a new detection box to the appropriate category.
        
        Args:
            box_data: Dictionary containing box information
                - x1, y1, x2, y2: coordinates
                - label: category label
                - conf: confidence (optional)
                - class: class index (optional)
            
        Returns:
            str: The ID assigned to the new box
        """
        try:
            # Validate required fields
            required_fields = ['x1', 'y1', 'x2', 'y2', 'label']
            for field in required_fields:
                if field not in box_data:
                    print(f"Missing required field: {field}")
                    return None
            
            # Create a copy of the box data
            new_box = box_data.copy()
            
            # Generate a unique ID for the new box
            label = new_box['label']
            if label == 'bar':
                new_id = f"bar_{self._next_index('bar')}"
                self.bars[new_id] = new_box
            elif label == 'uptail':
                new_id = f"uptail_{self._next_index('uptail')}"
                self.uptails[new_id] = new_box
            elif label == 'yaxis':
                new_id = f"yaxis_{self._next_index('yaxis')}"
                self.yaxis[new_id] = new_box
            elif label == 'xaxis':
                new_id = f"xaxis_{self._next_index('xaxis')}"
                self.xaxis[new_id] = new_box
            elif label == 'origin':
                new_id = f"origin_{self._next_index('origin')}"
                self.origins[new_id] = new_box
            elif label == 'ymax':
                new_id = f"ymax_{self._next_index('ymax')}"
                self.ymaxes[new_id] = new_box
            elif label == 'label':
                new_id = f"label_{self._next_index('label')}"
                self.labels[new_id] = new_box
            elif label == 'x_group':
                new_id = f"x_group_{self._next_index('x_group')}"
                self.x_groups[new_id] = new_box
            elif label == 'legend':
                new_id = f"legend_{self._next_index('legend')}"
                self.legends[new_id] = new_box
            elif label == 'legend_group':
                new_id = f"legend_group_{self._next_index('legend_group')}"
                self.legend_groups[new_id] = new_box
            else:
                print(f"Unknown label: {label}")
                return None
            
            # Set the ID in the box data
            new_box['id'] = new_id
            
            # Add to detections list for consistency
            if self.detections is None:
                self.detections = []
            self.detections.append(new_box)
            
            print(f"Added new box with ID: {new_id} to category: {label}")
            return new_id
            
        except Exception as e:
            print(f"Error adding new box: {e}")
            return None

    def remove_box(self, box_id: str) -> bool:
        """
        Remove a detection box from the data structures.
        
        Args:
            box_id: Unique identifier of the box to remove
            
        Returns:
            bool: True if the box was successfully removed
        """
        try:
            # Find the box in any of the category dictionaries
            box_data = None
            old_category = None
            
            # Check all category dictionaries to find the box
            for category_dict in self._category_dicts:
                if box_id in category_dict:
                    box_data = category_dict[box_id]
                    # Determine which category this was
                    if category_dict is self.bars:
                        old_category = 'bar'
                    elif category_dict is self.uptails:
                        old_category = 'uptail'
                    elif category_dict is self.yaxis:
                        old_category = 'yaxis'
                    elif category_dict is self.xaxis:
                        old_category = 'xaxis'
                    elif category_dict is self.origins:
                        old_category = 'origin'
                    elif category_dict is self.ymaxes:
                        old_category = 'ymax'
                    elif category_dict is self.labels:
                        old_category = 'label'
                    elif category_dict is self.x_groups:
                        old_category = 'x_group'
                    elif category_dict is self.legends:
                        old_category = 'legend'
                    elif category_dict is self.legend_groups:
                        old_category = 'legend_group'
                    break
            
            if not box_data:
                print(f"Box with ID {box_id} not found for removal")
                return False
            
            # Remove from the appropriate category dictionary
            if old_category == 'bar':
                del self.bars[box_id]
            elif old_category == 'uptail':
                del self.uptails[box_id]
            elif old_category == 'yaxis':
                del self.yaxis[box_id]
            elif old_category == 'xaxis':
                del self.xaxis[box_id]
            elif old_category == 'origin':
                del self.origins[box_id]
            elif old_category == 'ymax':
                del self.ymaxes[box_id]
            elif old_category == 'label':
                del self.labels[box_id]
            elif old_category == 'x_group':
                del self.x_groups[box_id]
            elif old_category == 'legend':
                del self.legends[box_id]
            elif old_category == 'legend_group':
                del self.legend_groups[box_id]
            
            # Remove from detections list if it exists
            if self.detections:
                self.detections = [det for det in self.detections if det.get('id') != box_id]
            
            print(f"Successfully removed box {box_id} from category {old_category}")
            return True
            
        except Exception as e:
            print(f"Error removing box {box_id}: {e}")
            return False


if __name__ == "__main__":
    # Simple manual test harness (adjust paths as needed)
    model_path = 'models/best.pt'
    class_names = ['label', 'ymax', 'origin', 'yaxis', 'bar', 'uptail', 'legend', 'legend_group', 'xaxis', 'x_group']
    analyzer = BarGraphAnalyzer(model_path, class_names, pytesseract_cmd='/opt/homebrew/bin/tesseract')
    # analyzer.detect_box('/path/to/image.png')
    # print(analyzer.calculate_heights())