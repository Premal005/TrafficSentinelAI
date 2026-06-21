"""
Scene Graph — Layer 3 of the HSUP Pipeline.

The crown jewel differentiator of TrafficSentinel AI. This module constructs
a relational graph from raw detections, modelling *who rides what*, *who wears
what*, *which plate belongs to which vehicle*, and every other spatial
relationship needed for downstream violation reasoning.

Architecture
────────────
    Detections (flat list)
          │
          ▼
    SceneGraph.build()
      ├── _associate_riders_to_vehicles()   → RIDES edges
      ├── _associate_drivers_to_vehicles()  → DRIVES edges
      ├── _associate_helmets_to_riders()    → WEARS / NOT_WEARS edges
      ├── _associate_plates_to_vehicles()   → HAS_PLATE edges
      ├── _compute_directions()             → FACING attributes
      └── _detect_lane_positions()          → IN_LANE attributes

    Rich query API  →  used by ViolationEngine (Layer 4)
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from config.settings import (
    EntityClass,
    SceneGraphConfig,
    Settings,
    SETTINGS,
    TWO_WHEELER_CLASSES,
    FOUR_WHEELER_CLASSES,
)

if TYPE_CHECKING:
    from core.entity_detector import Detection  # avoid circular import

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# RELATION TYPE CONSTANTS
# ══════════════════════════════════════════════════════════════

RIDES: str = "RIDES"
"""Person → two-wheeler relationship (motorcycles, bicycles)."""

DRIVES: str = "DRIVES"
"""Person → four-wheeler relationship (car, bus, truck)."""

WEARS: str = "WEARS"
"""Person → safety gear (helmet, seatbelt)."""

NOT_WEARS: str = "NOT_WEARS"
"""Person → missing safety gear (no_helmet, no_seatbelt)."""

HAS_PLATE: str = "HAS_PLATE"
"""Vehicle → license plate."""

IN_LANE: str = "IN_LANE"
"""Entity → lane assignment."""

FACING: str = "FACING"
"""Entity → direction of travel."""

NEAR_SIGNAL: str = "NEAR_SIGNAL"
"""Entity → traffic signal proximity."""

CROSSED_LINE: str = "CROSSED_LINE"
"""Entity → stop-line crossing event."""

PASSENGER_OF: str = "PASSENGER_OF"
"""Additional passenger → vehicle relationship (triple-riding etc.)."""


# ══════════════════════════════════════════════════════════════
# GRAPH DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class SceneNode:
    """
    A single entity in the scene graph.

    Attributes:
        node_id:      Unique identifier (e.g. ``"person_3"``).
        entity_class: Semantic class from :class:`EntityClass`.
        bbox:         Bounding box ``(x1, y1, x2, y2)`` in pixel coords.
        confidence:   Detection confidence ``[0, 1]``.
        track_id:     Persistent track ID from the tracker, if available.
        attributes:   Free-form extras — ``plate_text``, ``direction_vector``,
                      ``helmet_status``, ``lane_index``, etc.
    """

    node_id: str
    entity_class: EntityClass
    bbox: Tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties ──────────────────────────────────

    @property
    def center(self) -> Tuple[float, float]:
        """Centre point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Pixel area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class SceneEdge:
    """
    A directed relationship between two :class:`SceneNode` instances.

    Attributes:
        source_id:  ``node_id`` of the source node.
        target_id:  ``node_id`` of the target node.
        relation:   One of the relation-type constants (``RIDES``, ``WEARS``, …).
        confidence: How confident we are in this relationship ``[0, 1]``.
        metadata:   Auxiliary data — IoU score, distance, etc.
    """

    source_id: str
    target_id: str
    relation: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Args:
        box1: ``(x1, y1, x2, y2)`` for the first box.
        box2: ``(x1, y1, x2, y2)`` for the second box.

    Returns:
        IoU value in ``[0.0, 1.0]``.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_overlap_ratio(
    inner_box: Tuple[int, int, int, int],
    outer_box: Tuple[int, int, int, int],
) -> float:
    """
    Fraction of *inner_box* that lies inside *outer_box*.

    Useful for checking whether a helmet detection is contained within a
    person detection, or whether a plate is within a vehicle bbox.

    Args:
        inner_box: ``(x1, y1, x2, y2)`` — the box whose containment we measure.
        outer_box: ``(x1, y1, x2, y2)`` — the reference container box.

    Returns:
        Ratio in ``[0.0, 1.0]``.  ``1.0`` means *inner_box* is fully inside
        *outer_box*.
    """
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = max(0, inner_box[2] - inner_box[0]) * max(0, inner_box[3] - inner_box[1])
    return inter / inner_area if inner_area > 0 else 0.0


def is_above(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> bool:
    """
    Check whether the centre of *box1* is above the centre of *box2*.

    In image coordinates the y-axis points **downward**, so "above" means a
    **smaller** y-value.

    Args:
        box1: ``(x1, y1, x2, y2)`` — candidate "above" box.
        box2: ``(x1, y1, x2, y2)`` — candidate "below" box.

    Returns:
        ``True`` if *box1*'s centre-y is strictly less than *box2*'s centre-y.
    """
    cy1 = (box1[1] + box1[3]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return cy1 < cy2


def is_four_wheeler_occupant(
    person_bbox: Tuple[int, int, int, int],
    vehicle_bbox: Tuple[int, int, int, int],
) -> bool:
    """
    Check if a person's bounding box matches standard occupant/driver constraints
    for a four-wheeler/auto-rickshaw cabin.
    """
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox
    p_h = py2 - py1
    p_w = px2 - px1
    v_h = vy2 - vy1
    v_w = vx2 - vx1
    
    if v_h <= 0 or v_w <= 0:
        return False
        
    overlap = compute_overlap_ratio(person_bbox, vehicle_bbox)
    
    # Occupant must be mostly inside the vehicle bounding box
    if overlap < 0.40:
        return False
        
    # Occupant relative size checks
    height_ratio = p_h / v_h
    width_ratio = p_w / v_w
    if not (0.15 <= height_ratio <= 0.85):
        return False
    if not (0.10 <= width_ratio <= 0.65):
        return False
        
    # Occupant vertical positioning (cabin/windshield constraint)
    # The top of the occupant (head/hand) should be in the upper region or slightly above
    if not (vy1 - 0.35 * v_h <= py1 <= vy1 + 0.65 * v_h):
        return False
    # The bottom of the occupant should be in the upper/middle region
    if not (vy1 + 0.05 * v_h <= py2 <= vy1 + 0.95 * v_h):
        return False
        
    # Occupant horizontal positioning (within vehicle boundaries + margin)
    if not (vx1 - 0.15 * v_w <= px1 and px2 <= vx2 + 0.15 * v_w):
        return False
        
    return True


def is_head_riding_motorcycle(
    head_bbox: Tuple[int, int, int, int],
    motorcycle_bbox: Tuple[int, int, int, int],
    scene_graph: Optional[SceneGraph] = None,
    moto_node_id: Optional[str] = None,
) -> bool:
    """
    Check if a helmet/no_helmet detection box matches a rider's head position
    relative to the motorcycle's bounding box.
    """
    hx1, hy1, hx2, hy2 = head_bbox
    mx1, my1, mx2, my2 = motorcycle_bbox
    
    # Expand vertical and horizontal spans based on associated riders if available
    if scene_graph is not None and moto_node_id is not None:
        riders = scene_graph.get_riders_of(moto_node_id)
        for r in riders:
            rx1, ry1, rx2, ry2 = r.bbox
            mx1 = min(mx1, rx1)
            mx2 = max(mx2, rx2)
            my1 = min(my1, ry1)
            my2 = max(my2, ry2)
            
    mw = mx2 - mx1
    mh = my2 - my1
    
    if mw <= 0 or mh <= 0:
        return False
        
    # Horizontal alignment: The head must be largely within the motorcycle/riders horizontal span (with a 65% margin)
    left_limit = mx1 - 0.65 * mw
    right_limit = mx2 + 0.65 * mw
    if not (left_limit <= hx1 and hx2 <= right_limit):
        return False
        
    # Vertical alignment: The head must be above the motorcycle's bottom, and not too far above the motorcycle's top
    # The head bottom should be above the middle of the motorcycle
    if hy2 > my1 + 0.6 * mh:
        return False
    # The head top should not be more than 1.8 * mh above the top of the motorcycle (relaxed to handle foreshortening/children/tall occupants)
    if hy1 < my1 - 1.8 * mh:
        return False
        
    return True


def get_head_region(
    person_bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    Return the top 30 % of a person bounding box — the "head region".

    This is where we expect to find helmets (or the absence thereof).

    Args:
        person_bbox: ``(x1, y1, x2, y2)`` of the person.

    Returns:
        ``(x1, y1, x2, y_30pct)`` — the cropped head region.
    """
    x1, y1, x2, y2 = person_bbox
    height = y2 - y1
    head_y2 = int(y1 + height * 0.30)
    return (x1, y1, x2, head_y2)


def direction_from_vector(dx: float, dy: float) -> str:
    """
    Convert a displacement vector ``(dx, dy)`` to a compass direction.

    Uses image coordinates where **+y is down**, so a negative *dy* means
    the object is moving upward (→ ``NORTH``).

    Args:
        dx: Horizontal displacement (positive = rightward / EAST).
        dy: Vertical displacement   (positive = downward  / SOUTH).

    Returns:
        One of ``"NORTH"``, ``"SOUTH"``, ``"EAST"``, ``"WEST"``,
        ``"NORTHEAST"``, ``"NORTHWEST"``, ``"SOUTHEAST"``, ``"SOUTHWEST"``,
        or ``"STATIONARY"`` when the displacement is negligible.
    """
    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-6:
        return "STATIONARY"

    angle = math.degrees(math.atan2(-dy, dx))  # negate dy for image coords
    # Normalize to [0, 360)
    angle = angle % 360

    # 8-direction compass rose
    if 22.5 <= angle < 67.5:
        return "NORTHEAST"
    if 67.5 <= angle < 112.5:
        return "NORTH"
    if 112.5 <= angle < 157.5:
        return "NORTHWEST"
    if 157.5 <= angle < 202.5:
        return "WEST"
    if 202.5 <= angle < 247.5:
        return "SOUTHWEST"
    if 247.5 <= angle < 292.5:
        return "SOUTH"
    if 292.5 <= angle < 337.5:
        return "SOUTHEAST"
    return "EAST"


def _bbox_center_distance(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Euclidean distance between the centres of two bounding boxes."""
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return math.hypot(cx1 - cx2, cy1 - cy2)


# ══════════════════════════════════════════════════════════════
# SCENE GRAPH
# ══════════════════════════════════════════════════════════════

class SceneGraph:
    """
    Relational scene graph for a single video frame.

    Nodes represent detected entities (vehicles, persons, helmets, plates,
    traffic signals) and edges encode their spatial/semantic relationships
    (``RIDES``, ``WEARS``, ``HAS_PLATE``, …).

    Usage::

        sg = SceneGraph()
        sg.build(detections, trajectories, frame_shape)

        # Who's riding motorcycle_2?
        riders = sg.get_riders_of("motorcycle_2")

        # Any rider without a helmet?
        for rider in riders:
            status = sg.get_helmet_status(rider.node_id)
            if status == "no_helmet":
                print("Violation!")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialise the scene graph.

        Args:
            settings: Global settings instance.  Falls back to
                      :data:`SETTINGS` if ``None``.
        """
        self._settings: Settings = settings or SETTINGS
        self._sg_config: SceneGraphConfig = self._settings.scene_graph

        self.nodes: Dict[str, SceneNode] = {}
        self.edges: List[SceneEdge] = []

        self._node_counter: Dict[str, int] = {}  # per-class counter
        self.frame_shape: Tuple[int, int] | None = None

    # ──────────────────────────────────────────────────────────
    # BUILD
    # ──────────────────────────────────────────────────────────

    def build(
        self,
        detections: List[Any],
        trajectories: Dict[int, List] | None = None,
        frame_shape: Tuple[int, int] | None = None,
    ) -> "SceneGraph":
        """
        Construct the scene graph from a list of detections.

        This is the **main entry point** called once per frame.  It converts
        flat detections into rich relational structure by running all
        association heuristics.

        Args:
            detections:   List of ``Detection`` objects from the entity
                          detector.  Each must expose ``.entity_class``,
                          ``.bbox``, ``.confidence``, ``.track_id``, and
                          ``.metadata`` (dict).
            trajectories: Mapping ``track_id → list of (cx, cy)`` positions
                          over time, used for direction estimation.
            frame_shape:  ``(height, width)`` of the current frame, used for
                          coarse lane estimation.

        Returns:
            ``self`` — the populated :class:`SceneGraph` (allows chaining).
        """
        # Reset state
        self.nodes.clear()
        self.edges.clear()
        self._node_counter.clear()
        self.frame_shape = frame_shape

        if not detections:
            logger.debug("build() called with empty detections list.")
            return self

        # 1. Create nodes from detections
        for det in detections:
            try:
                node = self._detection_to_node(det)
                self.nodes[node.node_id] = node
            except Exception:
                logger.debug(
                    "Skipping non-traffic detection: %s", det.class_name if hasattr(det, 'class_name') else det
                )

        logger.info(
            "Scene graph: %d nodes created from %d detections.",
            len(self.nodes),
            len(detections),
        )

        # 2. Run association heuristics
        try:
            self._associate_persons_to_vehicles()
        except Exception:
            logger.error("Person-vehicle association failed.", exc_info=True)

        try:
            self._associate_helmets_to_riders()
        except Exception:
            logger.error("Helmet-rider association failed.", exc_info=True)

        try:
            self._associate_seatbelts_to_drivers()
        except Exception:
            logger.error("Seatbelt-driver association failed.", exc_info=True)

        try:
            self._associate_plates_to_vehicles()
        except Exception:
            logger.error("Plate–vehicle association failed.", exc_info=True)

        if trajectories:
            try:
                self._compute_directions(trajectories)
            except Exception:
                logger.error("Direction computation failed.", exc_info=True)

        if frame_shape:
            try:
                self._detect_lane_positions(frame_shape)
            except Exception:
                logger.error("Lane detection failed.", exc_info=True)

        # 3. Transitive Pruning: Keep only nodes connected to a vehicle, and infrastructure nodes
        vehicle_classes = TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW, EntityClass.TEMPO}
        infrastructure_classes = {
            EntityClass.STOP_LINE, EntityClass.TRAFFIC_LIGHT_RED,
            EntityClass.TRAFFIC_LIGHT_GREEN, EntityClass.TRAFFIC_LIGHT_YELLOW,
            EntityClass.LANE_MARKING
        }
        
        vehicle_nodes = [nid for nid, node in self.nodes.items() if node.entity_class in vehicle_classes]
        
        # If there are no vehicles at all, we keep everything (e.g. for cropped benchmark snippets)
        if not vehicle_nodes:
            pruned_nodes = self.nodes.copy()
        else:
            # Build an undirected adjacency list from current edges
            adj = {nid: set() for nid in self.nodes}
            for edge in self.edges:
                if edge.source_id in adj and edge.target_id in adj:
                    adj[edge.source_id].add(edge.target_id)
                    adj[edge.target_id].add(edge.source_id)
                    
            # BFS to find all nodes reachable from any vehicle node
            keep_node_ids = set()
            queue = list(vehicle_nodes)
            keep_node_ids.update(vehicle_nodes)
            
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in keep_node_ids:
                        keep_node_ids.add(neighbor)
                        queue.append(neighbor)
                        
            # Also keep infrastructure nodes regardless of connection
            for nid, node in self.nodes.items():
                if node.entity_class in infrastructure_classes:
                    keep_node_ids.add(nid)
                    
            # Prune nodes
            pruned_nodes = {}
            for nid, node in self.nodes.items():
                if nid in keep_node_ids:
                    pruned_nodes[nid] = node
                else:
                    logger.info("Pruning isolated node from scene graph: %s (%s)", nid, node.entity_class.value)
                    
            # Prune edges that refer to pruned nodes
            self.edges = [
                e for e in self.edges
                if e.source_id in keep_node_ids and e.target_id in keep_node_ids
            ]
            
        self.nodes = pruned_nodes

        logger.info(
            "Scene graph built: %d nodes, %d edges.",
            len(self.nodes),
            len(self.edges),
        )
        return self

    # ──────────────────────────────────────────────────────────
    # NODE CONSTRUCTION
    # ──────────────────────────────────────────────────────────

    def _detection_to_node(self, det: Any) -> SceneNode:
        """
        Convert a single ``Detection`` into a :class:`SceneNode`.

        Args:
            det: A detection object with ``.entity_class``, ``.bbox``,
                 ``.confidence``, ``.track_id``, ``.metadata``.

        Returns:
            A freshly created :class:`SceneNode`.
        """
        entity_class: EntityClass = det.entity_class
        if entity_class is None:
            raise ValueError(f"Detection class '{det.class_name}' is not mapped to an EntityClass.")
        class_name = entity_class.value

        # Generate a deterministic node id
        count = self._node_counter.get(class_name, 0)
        self._node_counter[class_name] = count + 1
        node_id = f"{class_name}_{count}"

        bbox = tuple(int(c) for c in det.bbox)

        attributes: Dict[str, Any] = {}
        if hasattr(det, "metadata") and isinstance(det.metadata, dict):
            attributes.update(det.metadata)

        return SceneNode(
            node_id=node_id,
            entity_class=entity_class,
            bbox=bbox,  # type: ignore[arg-type]
            confidence=float(det.confidence),
            track_id=getattr(det, "track_id", None),
            attributes=attributes,
        )

    # ──────────────────────────────────────────────────────────
    # ASSOCIATION HEURISTICS
    # ──────────────────────────────────────────────────────────

    def _associate_persons_to_vehicles(self) -> None:
        """
        Link person nodes to the best overlapping vehicle (2-wheeler or 4-wheeler).
        Creates RIDES edges for 2-wheelers and DRIVES edges for 4-wheelers.
        """
        persons = self._nodes_of_classes({EntityClass.PERSON, EntityClass.RIDER, EntityClass.DRIVER})
        all_vehicle_classes = TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW, EntityClass.TEMPO}
        vehicles = self._nodes_of_classes(all_vehicle_classes)

        for person in persons:
            best_vehicle: Optional[SceneNode] = None
            best_score: float = 0.0

            for vehicle in vehicles:
                iou = compute_iou(person.bbox, vehicle.bbox)
                overlap = compute_overlap_ratio(person.bbox, vehicle.bbox)
                score = max(iou, overlap)

                is_two_wheeler = vehicle.entity_class in TWO_WHEELER_CLASSES
                is_riding_heur = False
                
                if is_two_wheeler:
                    px1, py1, px2, py2 = person.bbox
                    vx1, vy1, vx2, vy2 = vehicle.bbox
                    p_w = px2 - px1
                    p_h = py2 - py1
                    
                    horiz_overlap = max(0, min(px2, vx2) - max(px1, vx1))
                    horiz_ratio = horiz_overlap / p_w if p_w > 0 else 0.0
                    
                    # Vertical distance from bottom of person to top of vehicle
                    vert_dist_ratio = (vy1 - py2) / p_h if p_h > 0 else 999.0
                    
                    # If they align horizontally and are stacked vertically, they are riding!
                    is_riding_heur = (horiz_ratio >= 0.40) and (-0.50 <= vert_dist_ratio <= 0.25)
                    
                    if is_riding_heur:
                        score = max(score, 0.50)

                thresh = self._sg_config.rider_vehicle_iou_thresh if is_two_wheeler else self._sg_config.person_vehicle_iou_thresh

                if score < thresh:
                    continue

                if is_two_wheeler and not is_riding_heur:
                    # Height ratio check for two-wheelers
                    person_h = person.bbox[3] - person.bbox[1]
                    vehicle_h = vehicle.bbox[3] - vehicle.bbox[1]
                    if vehicle_h > 0 and (person_h / vehicle_h) < 0.35:
                        continue
                    # Vertical check
                    if not is_above(person.bbox, vehicle.bbox) and overlap < 0.5:
                        continue
                
                if not is_two_wheeler:
                    if not is_four_wheeler_occupant(person.bbox, vehicle.bbox):
                        continue
                
                if score > best_score:
                    best_score = score
                    best_vehicle = vehicle

            if best_vehicle is not None:
                relation = RIDES if best_vehicle.entity_class in TWO_WHEELER_CLASSES else DRIVES
                self.edges.append(
                    SceneEdge(
                        source_id=person.node_id,
                        target_id=best_vehicle.node_id,
                        relation=relation,
                        confidence=min(person.confidence, best_vehicle.confidence),
                        metadata={"iou": round(best_score, 4)},
                    )
                )
                logger.debug(
                    "%s edge: %s -> %s (score=%.3f)",
                    relation,
                    person.node_id,
                    best_vehicle.node_id,
                    best_score,
                )

        # Pass 2: Associate remaining unassociated persons if they overlap with an already associated rider
        associated_people = {edge.source_id: edge.target_id for edge in self.edges if edge.relation in (RIDES, DRIVES)}
        unassociated_persons = [p for p in persons if p.node_id not in associated_people]
        
        for person in unassociated_persons:
            best_associated_person = None
            best_overlap = 0.0
            for assoc_id, vehicle_id in associated_people.items():
                assoc_node = self.nodes.get(assoc_id)
                if assoc_node is None:
                    continue
                # Only propagate for two-wheelers (motorcycles/bicycles)
                veh_node = self.nodes.get(vehicle_id)
                if veh_node is None or veh_node.entity_class not in TWO_WHEELER_CLASSES:
                    continue
                
                # Check overlap/IoU between the unassociated person and the associated rider
                overlap = compute_overlap_ratio(person.bbox, assoc_node.bbox)
                iou = compute_iou(person.bbox, assoc_node.bbox)
                max_overlap = max(overlap, iou)
                
                # If they overlap significantly, they are riding the same vehicle!
                # 0.15 is a standard, safe threshold for horizontal/vertical person overlaps on motorcycles
                if max_overlap > best_overlap and max_overlap >= 0.15:
                    best_overlap = max_overlap
                    best_associated_person = assoc_node
                    
            if best_associated_person is not None:
                vehicle_id = associated_people[best_associated_person.node_id]
                vehicle_node = self.nodes[vehicle_id]
                self.edges.append(
                    SceneEdge(
                        source_id=person.node_id,
                        target_id=vehicle_id,
                        relation=RIDES,
                        confidence=min(person.confidence, vehicle_node.confidence),
                        metadata={"iou": round(best_overlap, 4), "propagated": True}
                    )
                )
                logger.info(
                    "Associated person %s to vehicle %s via overlap with rider %s (overlap=%.3f)",
                    person.node_id,
                    vehicle_id,
                    best_associated_person.node_id,
                    best_overlap
                )

    def _associate_seatbelts_to_drivers(self) -> None:
        """
        Link seatbelt / no-seatbelt detections to the nearest driver.
        """
        persons = self._nodes_of_classes({EntityClass.PERSON, EntityClass.DRIVER})
        seatbelts = self._nodes_of_classes({EntityClass.SEATBELT})
        no_seatbelts = self._nodes_of_classes({EntityClass.NO_SEATBELT})

        for gear_node in seatbelts:
            best_person, best_score = self._find_best_person_for_seatbelt(gear_node, persons)
            if best_person:
                self.edges.append(
                    SceneEdge(
                        source_id=best_person.node_id, 
                        target_id=gear_node.node_id, 
                        relation=WEARS, 
                        confidence=min(best_person.confidence, gear_node.confidence), 
                        metadata={"overlap": round(best_score, 4)}
                    )
                )
                best_person.attributes["seatbelt_status"] = "seatbelt"

        for gear_node in no_seatbelts:
            best_person, best_score = self._find_best_person_for_seatbelt(gear_node, persons)
            if best_person:
                self.edges.append(
                    SceneEdge(
                        source_id=best_person.node_id, 
                        target_id=gear_node.node_id, 
                        relation=NOT_WEARS, 
                        confidence=min(best_person.confidence, gear_node.confidence), 
                        metadata={"overlap": round(best_score, 4)}
                    )
                )
                if best_person.attributes.get("seatbelt_status") != "seatbelt":
                    best_person.attributes["seatbelt_status"] = "no_seatbelt"

    def _find_best_person_for_seatbelt(self, gear_node: SceneNode, persons: List[SceneNode]) -> Tuple[Optional[SceneNode], float]:
        best_person = None
        best_score = 0.0
        for person in persons:
            overlap = compute_overlap_ratio(gear_node.bbox, person.bbox)
            if overlap > best_score and overlap > 0.5:
                best_score = overlap
                best_person = person
        return best_person, best_score

    def _associate_helmets_to_riders(self) -> None:
        """
        Link helmet / no-helmet detections to the nearest person's head
        region using greedy matching.
        """
        persons = self._nodes_of_classes(
            {EntityClass.PERSON, EntityClass.RIDER}
        )
        helmets = self._nodes_of_classes({EntityClass.HELMET})
        no_helmets = self._nodes_of_classes({EntityClass.NO_HELMET})

        proximity = self._sg_config.helmet_person_proximity

        # 1. Collect all possible candidates with their scores
        all_candidates = []
        for gear_node in helmets + no_helmets:
            for person in persons:
                head = get_head_region(person.bbox)
                overlap = compute_overlap_ratio(gear_node.bbox, head)

                # Check raw distance as fallback
                dist = _bbox_center_distance(gear_node.bbox, head)
                person_height = person.bbox[3] - person.bbox[1]
                relative_dist = dist / person_height if person_height > 0 else float("inf")

                score = 0.0
                if overlap >= proximity:
                    score = 1.0 + overlap
                elif overlap == 0.0 and relative_dist < 0.5:
                    score = 1.0 - relative_dist

                if score > 0.0:
                    all_candidates.append({
                        "gear": gear_node,
                        "person": person,
                        "score": score,
                        "overlap": overlap
                    })

        # 2. Match greedily by score descending
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        matched_persons = set()
        matched_gears = set()

        for cand in all_candidates:
            gear = cand["gear"]
            person = cand["person"]

            if gear.node_id in matched_gears or person.node_id in matched_persons:
                continue

            matched_persons.add(person.node_id)
            matched_gears.add(gear.node_id)

            relation = WEARS if gear.entity_class == EntityClass.HELMET else NOT_WEARS
            status_str = "helmet" if gear.entity_class == EntityClass.HELMET else "no_helmet"

            self.edges.append(
                SceneEdge(
                    source_id=person.node_id,
                    target_id=gear.node_id,
                    relation=relation,
                    confidence=min(person.confidence, gear.confidence),
                    metadata={"overlap": round(cand["overlap"], 4)}
                )
            )
            person.attributes["helmet_status"] = status_str

        # 3. Synthesize occluded person nodes for orphan helmets/no-helmets on motorcycles
        associated_gear_ids = {edge.target_id for edge in self.edges if edge.relation in (WEARS, NOT_WEARS)}
        orphan_gears = [g for g in helmets + no_helmets if g.node_id not in associated_gear_ids]
        
        motorcycles = self._nodes_of_classes(TWO_WHEELER_CLASSES)
        
        for gear in orphan_gears:
            # Find matching motorcycle
            best_moto = None
            for moto in motorcycles:
                if is_head_riding_motorcycle(gear.bbox, moto.bbox, self, moto.node_id):
                    best_moto = moto
                    break
                    
            if best_moto is not None:
                hx1, hy1, hx2, hy2 = gear.bbox
                # Synthesize a person node by expanding head box downwards
                p_h = hy2 - hy1
                p_bbox = (hx1, hy1, hx2, hy1 + 4 * p_h)
                
                # Check that y-coords are within reasonable range
                if self.frame_shape is not None:
                    img_h, img_w = self.frame_shape[:2]
                    p_bbox = (
                        max(0, min(img_w, hx1)),
                        max(0, min(img_h, hy1)),
                        max(0, min(img_w, hx2)),
                        max(0, min(img_h, hy1 + 4 * p_h))
                    )
                
                import uuid
                p_id = f"person_syn_{uuid.uuid4().hex[:4]}"
                
                syn_person = SceneNode(
                    node_id=p_id,
                    entity_class=EntityClass.PERSON,
                    bbox=p_bbox,
                    confidence=gear.confidence,
                    attributes={"synthesized": True}
                )
                
                self.nodes[p_id] = syn_person
                
                # Add RIDES edge
                self.edges.append(
                    SceneEdge(
                        source_id=p_id,
                        target_id=best_moto.node_id,
                        relation=RIDES,
                        confidence=min(syn_person.confidence, best_moto.confidence),
                        metadata={"synthesized": True}
                    )
                )
                
                # Add WEARS / NOT_WEARS edge
                relation = WEARS if gear.entity_class == EntityClass.HELMET else NOT_WEARS
                status_str = "helmet" if gear.entity_class == EntityClass.HELMET else "no_helmet"
                
                self.edges.append(
                    SceneEdge(
                        source_id=p_id,
                        target_id=gear.node_id,
                        relation=relation,
                        confidence=gear.confidence,
                        metadata={"overlap": 1.0}
                    )
                )
                syn_person.attributes["helmet_status"] = status_str
                
                logger.info(f"Synthesized occupant {p_id} for orphan {status_str} node on motorcycle {best_moto.node_id}")

    def _find_closest_head(
        self,
        gear_node: SceneNode,
        persons: List[SceneNode],
        min_proximity: float,
    ) -> Tuple[Optional[SceneNode], float]:
        """
        Find the person whose head region best overlaps *gear_node*.

        Args:
            gear_node:     The helmet or no-helmet node.
            persons:       Candidate person nodes.
            min_proximity: Minimum overlap ratio to consider a match.

        Returns:
            ``(best_person, best_overlap)`` or ``(None, 0.0)`` if no match.
        """
        best_person: Optional[SceneNode] = None
        best_score: float = 0.0
        best_overlap_val: float = 0.0

        for person in persons:
            head = get_head_region(person.bbox)
            overlap = compute_overlap_ratio(gear_node.bbox, head)

            # Also check raw distance as fallback
            dist = _bbox_center_distance(gear_node.bbox, head)
            person_height = person.bbox[3] - person.bbox[1]
            relative_dist = dist / person_height if person_height > 0 else float("inf")

            if overlap >= min_proximity:
                # Prioritize overlap: score ranges [1.0, 2.0]
                score = 1.0 + overlap
                if score > best_score:
                    best_score = score
                    best_overlap_val = overlap
                    best_person = person
            elif overlap == 0.0 and relative_dist < 0.5:
                # Fallback to distance only if no overlap: score ranges [0.5, 1.0]
                score = 1.0 - relative_dist
                if score > best_score:
                    best_score = score
                    best_overlap_val = 0.0
                    best_person = person

        return best_person, best_overlap_val

    def _associate_plates_to_vehicles(self) -> None:
        """
        Link license-plate nodes to the nearest vehicle.

        A plate is assigned to the vehicle whose bbox has the highest
        overlap with the plate bbox.  Typically plates are *inside* the
        vehicle bbox, so ``compute_overlap_ratio(plate, vehicle)`` is the
        primary metric.
        """
        plates = self._nodes_of_classes({EntityClass.LICENSE_PLATE})
        all_vehicle_classes = TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {
            EntityClass.AUTO_RICKSHAW
        }
        vehicles = self._nodes_of_classes(all_vehicle_classes)

        iou_thresh = self._sg_config.plate_vehicle_iou_thresh

        for plate in plates:
            best_vehicle: Optional[SceneNode] = None
            best_score: float = 0.0

            for vehicle in vehicles:
                # YOLO vehicle bounding boxes sometimes cut off the lower bumper.
                # Expand the bounding box by 10% horizontally and 20% vertically downward
                # to ensure plates on the bumper are correctly associated.
                vx1, vy1, vx2, vy2 = vehicle.bbox
                vw = max(0, vx2 - vx1)
                vh = max(0, vy2 - vy1)
                expand_down = 0.45 if vehicle.entity_class in FOUR_WHEELER_CLASSES else 0.25
                expanded_vehicle_bbox = (
                    int(vx1 - 0.15 * vw),
                    int(vy1 - 0.1 * vh),
                    int(vx2 + 0.15 * vw),
                    int(vy2 + expand_down * vh)
                )
                
                overlap = compute_overlap_ratio(plate.bbox, expanded_vehicle_bbox)
                if overlap < iou_thresh:
                    continue
                if overlap > best_score:
                    best_score = overlap
                    best_vehicle = vehicle

            if best_vehicle is not None:
                self.edges.append(
                    SceneEdge(
                        source_id=best_vehicle.node_id,
                        target_id=plate.node_id,
                        relation=HAS_PLATE,
                        confidence=min(
                            best_vehicle.confidence, plate.confidence
                        ),
                        metadata={
                            "overlap": round(best_score, 4),
                            "plate_text": plate.attributes.get("plate_text", ""),
                        },
                    )
                )
                logger.debug(
                    "HAS_PLATE edge: %s → %s",
                    best_vehicle.node_id,
                    plate.node_id,
                )
            else:
                logger.debug(
                    "Plate %s could not be associated with any vehicle.",
                    plate.node_id,
                )

    def _compute_directions(
        self, trajectories: Dict[int, List]
    ) -> None:
        """
        Compute movement direction from tracker trajectories.

        For each node that has a ``track_id``, look up its trajectory and
        compute a smoothed displacement vector over the last
        ``direction_smoothing_window`` positions.

        Args:
            trajectories: ``{track_id: [(cx, cy), …]}`` position history.
        """
        min_len = self._sg_config.min_trajectory_length
        window = self._sg_config.direction_smoothing_window

        for node in self.nodes.values():
            if node.track_id is None:
                continue
            traj = trajectories.get(node.track_id)
            if not traj or len(traj) < min_len:
                continue

            # Use the last `window` points for smoothed direction
            recent = traj[-window:]
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            direction = direction_from_vector(dx, dy)

            node.attributes["direction"] = direction
            node.attributes["direction_vector"] = (round(dx, 2), round(dy, 2))

            logger.debug(
                "Direction for %s (track %d): %s",
                node.node_id,
                node.track_id,
                direction,
            )

    def _detect_lane_positions(
        self, frame_shape: Tuple[int, int]
    ) -> None:
        """
        Rough lane assignment based on horizontal position within the frame.

        Divides the frame width into equal-width lanes and assigns each
        vehicle to a lane index.  This is a *coarse* heuristic — a
        production system would use detected lane markings or a calibrated
        perspective transform.

        Args:
            frame_shape: ``(height, width)`` of the frame.
        """
        _, frame_width = frame_shape
        if frame_width <= 0:
            return

        num_lanes = 4  # default assumption
        lane_width = frame_width / num_lanes

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )
        vehicles = self._nodes_of_classes(all_vehicle_classes)

        for vehicle in vehicles:
            cx = vehicle.center[0]
            lane_index = min(int(cx / lane_width), num_lanes - 1)
            vehicle.attributes["lane_index"] = lane_index

            logger.debug(
                "Lane assignment: %s → lane %d", vehicle.node_id, lane_index
            )

    # ──────────────────────────────────────────────────────────
    # QUERY API
    # ──────────────────────────────────────────────────────────

    def get_riders_of(self, vehicle_node_id: str) -> List[SceneNode]:
        """
        Return all persons riding a given vehicle.

        Args:
            vehicle_node_id: ``node_id`` of the vehicle.

        Returns:
            List of person/rider :class:`SceneNode` instances linked via
            ``RIDES`` or ``PASSENGER_OF`` edges.
        """
        rider_ids: List[str] = []
        for edge in self.edges:
            if edge.target_id == vehicle_node_id and edge.relation in (
                RIDES,
                PASSENGER_OF,
            ):
                rider_ids.append(edge.source_id)
        return [self.nodes[rid] for rid in rider_ids if rid in self.nodes]

    def get_vehicle_of(self, person_node_id: str) -> Optional[SceneNode]:
        """
        Return the vehicle a person is riding or driving.

        Args:
            person_node_id: ``node_id`` of the person.

        Returns:
            The :class:`SceneNode` of the vehicle, or ``None``.
        """
        for edge in self.edges:
            if edge.source_id == person_node_id and edge.relation in (
                RIDES,
                DRIVES,
                PASSENGER_OF,
            ):
                return self.nodes.get(edge.target_id)
        return None

    def get_helmet_status(self, person_node_id: str) -> Optional[str]:
        """
        Return the helmet status of a person.

        Args:
            person_node_id: ``node_id`` of the person.

        Returns:
            ``"helmet"``, ``"no_helmet"``, or ``None`` if unknown.
        """
        node = self.nodes.get(person_node_id)
        if node is None:
            return None

        # Check attribute first (set during helmet association)
        status = node.attributes.get("helmet_status")
        if status is not None:
            return status

        # Fallback: walk edges
        for edge in self.edges:
            if edge.source_id == person_node_id:
                if edge.relation == WEARS:
                    target = self.nodes.get(edge.target_id)
                    if target and target.entity_class == EntityClass.HELMET:
                        return "helmet"
                if edge.relation == NOT_WEARS:
                    target = self.nodes.get(edge.target_id)
                    if target and target.entity_class == EntityClass.NO_HELMET:
                        return "no_helmet"
        return None

    def get_plate_of(self, vehicle_node_id: str) -> Optional[SceneNode]:
        """
        Return the license plate associated with a vehicle.

        Args:
            vehicle_node_id: ``node_id`` of the vehicle.

        Returns:
            The plate :class:`SceneNode`, or ``None``.
        """
        for edge in self.edges:
            if (
                edge.source_id == vehicle_node_id
                and edge.relation == HAS_PLATE
            ):
                return self.nodes.get(edge.target_id)
        return None

    def get_passenger_count(self, vehicle_node_id: str) -> int:
        """
        Count the number of persons on/in a vehicle.

        Includes all ``RIDES``, ``DRIVES``, and ``PASSENGER_OF`` edges.

        Args:
            vehicle_node_id: ``node_id`` of the vehicle.

        Returns:
            Integer count of associated persons.
        """
        count = 0
        for edge in self.edges:
            if edge.target_id == vehicle_node_id and edge.relation in (
                RIDES,
                DRIVES,
                PASSENGER_OF,
            ):
                count += 1
        return count

    def get_direction(self, vehicle_node_id: str) -> Optional[str]:
        """
        Return the compass direction of a vehicle's movement.

        Args:
            vehicle_node_id: ``node_id`` of the vehicle.

        Returns:
            Compass direction string (``"NORTH"``, ``"SOUTH"``, …) or
            ``None`` if direction data is unavailable.
        """
        node = self.nodes.get(vehicle_node_id)
        if node is None:
            return None
        return node.attributes.get("direction")

    def query(
        self,
        source_class: EntityClass | None = None,
        relation: str | None = None,
        target_class: EntityClass | None = None,
    ) -> List[Tuple[SceneNode, SceneEdge, SceneNode]]:
        """
        General-purpose graph query — returns matching triplets.

        All parameters are optional filters; omitting a parameter matches
        everything.

        Args:
            source_class: Filter source node by :class:`EntityClass`.
            relation:     Filter by relation type string.
            target_class: Filter target node by :class:`EntityClass`.

        Returns:
            List of ``(source_node, edge, target_node)`` triplets.
        """
        results: List[Tuple[SceneNode, SceneEdge, SceneNode]] = []

        for edge in self.edges:
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            if src is None or tgt is None:
                continue

            if source_class is not None and src.entity_class != source_class:
                continue
            if relation is not None and edge.relation != relation:
                continue
            if target_class is not None and tgt.entity_class != target_class:
                continue

            results.append((src, edge, tgt))

        return results

    # ──────────────────────────────────────────────────────────
    # SERIALISATION
    # ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the scene graph to a JSON-safe dictionary.

        Returns:
            Dictionary with ``"nodes"`` and ``"edges"`` keys, suitable for
            ``json.dumps()``.
        """
        return {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "entity_class": n.entity_class.value,
                    "bbox": list(n.bbox),
                    "confidence": round(n.confidence, 4),
                    "track_id": n.track_id,
                    "attributes": n.attributes,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "confidence": round(e.confidence, 4),
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
        }

    def to_networkx(self) -> Any:
        """
        Convert the scene graph to a NetworkX ``DiGraph``.

        Each node carries its :class:`SceneNode` attributes; each edge
        carries its :class:`SceneEdge` relation and metadata.

        Returns:
            ``networkx.DiGraph`` instance.

        Raises:
            ImportError: If NetworkX is not installed.
        """
        try:
            import networkx as nx  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "NetworkX is required for to_networkx(). "
                "Install with: pip install networkx"
            )
            raise

        G = nx.DiGraph()

        for nid, node in self.nodes.items():
            G.add_node(
                nid,
                entity_class=node.entity_class.value,
                bbox=node.bbox,
                confidence=node.confidence,
                track_id=node.track_id,
                **node.attributes,
            )

        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation,
                confidence=edge.confidence,
                **edge.metadata,
            )

        return G

    # ──────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _nodes_of_classes(
        self, classes: set[EntityClass] | frozenset[EntityClass]
    ) -> List[SceneNode]:
        """
        Return all nodes whose ``entity_class`` is in *classes*.

        Args:
            classes: Set of :class:`EntityClass` values to filter by.

        Returns:
            Filtered list of :class:`SceneNode`.
        """
        return [n for n in self.nodes.values() if n.entity_class in classes]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SceneGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
        )
