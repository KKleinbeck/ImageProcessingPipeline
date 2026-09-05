"""Utility methods to generate mock image data."""

import numpy as np
from pydantic import BaseModel, Field
from matplotlib.path import Path
from scipy.signal import convolve2d


class BlobConfig(BaseModel):
  n_blobs: int = 64
  "Max number of blobs to generate"
  amp_range: tuple[float, float] = (0.5, 1.0)
  "Amplitude range of each blob"
  sigma_range: tuple[float, float] = (3.0, 9.0)
  "Gaussian width of each blob"


class BackgroundConfig(BaseModel):
  degree: int = 3
  "Degree of the polynomial (inclusive)."
  max_amplitude: float = 0.1
  "Largest possible amplitude of the degree 1 polynomial"


class StructureConfig(BaseModel):
  structure_grid: tuple[int, int] = 8, 8
  "Shape of the (small) grid to place nodes on, as (height, width)."
  n_structures: int = 8
  "Number of distinct nodes to sample from the grid's perimeter."


class NoiseConfig(BaseModel):
  """Lightweight configuration for Perlin noise generation."""

  amplitude: float = Field(0.1, gt=0, description="Overall scale of the noise output.")
  "Scales the overall noise output; final values roughly span [0, 2*amplitude]."
  noise_level: float = Field(
    8.0,
    gt=0,
    description=(
      "Approximate number of noise periods across the frame's shorter side; higher = finer/noisier, lower = smoother."
    ),
  )


class SampleDataConfig(BaseModel):
  cell_size: tuple[int, int] = (256, 256)
  "Pixel size of the cell (blob containing region)"
  frame_size: tuple[int, int] = (512, 512)
  "Pixel size of the overall frame"

  blob_config: BlobConfig = BlobConfig()
  background_config: BackgroundConfig | None = BackgroundConfig()
  structure_config: StructureConfig | None = StructureConfig()
  noise_config: NoiseConfig | None = NoiseConfig()

  seed: int | None = None
  "Seed for the ranom number generator."


def _widen_lines(frame: np.ndarray, kernel_1d: np.ndarray | None = None) -> np.ndarray:
  """Blur a binary line drawing to widen the lines.

  Convolves `frame` with a separable 2D kernel built as the outer
  (cartesian) product of `kernel_1d` with itself, then clips the result
  to a maximum of 1 (line intersections/corners otherwise sum to values
  greater than 1).

  Parameters
  ----------
  frame : np.ndarray
    2D array containing single-pixel-wide lines (e.g. the output of
    repeated `_draw_line` calls), typically valued 0 or 1.
  kernel_1d : Sequence[float] or None, optional
    1D kernel used to build the separable 2D blur kernel via
    ``np.outer(kernel_1d, kernel_1d)``. If None (default), uses
    ``[0.25, 0.5, 0.75, 1, 1, 1, 1, 0.75, 0.5, 0.25]``.

  Returns
  -------
  np.ndarray
    The blurred (widened) line array, same shape as `frame`, with
    values clipped to the range [0, 1].

  """
  if kernel_1d is None:
    kernel_1d = np.array([0.25, 0.5, 0.75, 1, 1, 1, 1, 0.75, 0.5, 0.25])
  else:
    kernel_1d = np.asarray(kernel_1d, dtype=float)

  kernel_2d = np.outer(kernel_1d, kernel_1d)
  kernel_2d = kernel_2d / kernel_1d.sum()  # Normalise for lines

  frame_blurred = convolve2d(frame, kernel_2d, mode="same")
  frame_blurred[frame_blurred > 1.0] = 1.0  # Corners will get larger values, force them down
  return frame_blurred


def _draw_line(p0: tuple[int, int], p1: tuple[int, int], img: np.ndarray) -> None:
  """Draw a single-pixel-wide line between two points onto an array.

  Uses Bresenham's line algorithm. Anti-aliasing is intentionally
  ignored, so the resulting line is exactly one pixel wide everywhere.

  Parameters
  ----------
  p0 : tuple[int, int]
    Start point of the line, as (x, y) i.e. (column, row).
  p1 : tuple[int, int]
    End point of the line, as (x, y) i.e. (column, row).
  img : np.ndarray
    2D array the line is drawn onto. Modified in place; pixels on the
    line are set to ``1.0``.

  Returns
  -------
  None
    `img` is modified in place; nothing is returned.

  """
  x0, y0 = int(round(p0[0])), int(round(p0[1]))
  x1, y1 = int(round(p1[0])), int(round(p1[1]))
  dx = abs(x1 - x0)
  dy = -abs(y1 - y0)
  sx = 1 if x0 < x1 else -1
  sy = 1 if y0 < y1 else -1
  err = dx + dy
  while True:
    img[y0, x0] = 1.0
    if x0 == x1 and y0 == y1:
      break
    e2 = 2 * err
    if e2 >= dy:
      err += dy
      x0 += sx
    if e2 <= dx:
      err += dx
      y0 += sy


def generate_gaussian_field(config: SampleDataConfig, max_attempts_per_gaussian: int = 500) -> np.ndarray:
  """Build an NxN array of zeros and add n_gaussians 2D Gaussians to it.

  Amplitudes and sigmas are drawn uniformly from `amp_range` /
  `sigma_range`. Centers are drawn uniformly from the sub-range
  ``[30, N - 30]`` in each dimension (keeping Gaussians away from the
  array border), but rejected (re-sampled) if the distance to any
  already-placed Gaussian's center is smaller than the sum of the two
  standard deviations (``sigma_i + sigma_j``).

  Parameters
  ----------
  config
    Configuration for the data generation
  max_attempts_per_gaussian : int, optional
    Maximum number of resample attempts per Gaussian before giving up
    on placing the full `n_gaussians`. Default is 500.

  Returns
  -------
  np.ndarray
    Array of shape (N, N) containing the sum of all successfully
    placed Gaussians. If fewer than `n_gaussians` could be placed
    within the attempt budget, a warning is printed and the array
    contains only the ones that were placed.

  """
  rng = np.random.default_rng(config.seed)
  field = np.zeros(config.cell_size)
  y_idx, x_idx = np.mgrid[0 : config.cell_size[0], 0 : config.cell_size[1]]

  centers, sigmas, amps = [], [], []
  placed, attempts = 0, 0
  max_total_attempts = max_attempts_per_gaussian * config.blob_config.n_blobs

  while placed < config.blob_config.n_blobs and attempts < max_total_attempts:
    attempts += 1
    amp: float = rng.uniform(*config.blob_config.amp_range)
    sigma: float = rng.uniform(*config.blob_config.sigma_range)
    cx: float = rng.uniform(30, config.cell_size[0] - 30)
    cy: float = rng.uniform(30, config.cell_size[1] - 30)

    # separation constraint: distance >= sigma_i + sigma_j
    distance_is_okay: bool = all(np.hypot(cx - ox, cy - oy) >= sigma + osig for (ox, oy), osig in zip(centers, sigmas))

    if distance_is_okay:
      centers.append((cx, cy))
      sigmas.append(sigma)
      amps.append(amp)
      field += amp * np.exp(-(((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (2 * sigma**2)))
      placed += 1

  if placed < config.blob_config.n_blobs:
    print(
      f"Warning: only placed {placed}/{config.blob_config.n_blobs} Gaussians "
      f"before hitting the attempt limit ({attempts} attempts)."
    )

  return field


def add_background(array: np.ndarray, config: SampleDataConfig) -> np.ndarray:
  """Add a random polynomial background to a 2D array.

  Builds a full bivariate polynomial from every monomial ``x**a * y**b``
  with total degree ``a + b`` ranging from 1 to `degree` (the constant
  term is excluded). For each total degree ``d``, all ``d + 1`` monomials
  of that degree (``x**d``, ``x**(d-1) * y``, ..., ``y**d``) get an
  independently-sampled coefficient.

  - The origin is at the center of the pixel grid.
  - x, y are normalised by the grid's half-width/half-height, so each
    pure term (``x**d`` or ``y**d``) individually reaches amplitude 1 at
    the array edges.
  - Each monomial of total degree ``d`` gets a coefficient drawn from
    ``Uniform(0, shrink_factor * max_amp / d)``, where
    ``max_amp = max(|array|)``. Dividing by ``d`` damps higher-order
    terms.
  - The background is added to `array`, and the result is shifted so
    its minimum is exactly 0.

  Parameters
  ----------
  array : np.ndarray
    2D input array to add the background to.
  config
    Configuration for the data generation

  Returns
  -------
  np.ndarray
    `array` plus the sampled polynomial background, shifted so that
    the result's minimum value is exactly 0. Same shape as `array`.

  """
  assert config.background_config is not None

  rng = np.random.default_rng(config.seed)
  array = np.asarray(array, dtype=float)
  ny, nx = array.shape

  half_width = (nx - 1) / 2.0
  half_height = (ny - 1) / 2.0

  x = (np.arange(nx) - half_width) / half_width  # in [-1, 1] at edges
  y = (np.arange(ny) - half_height) / half_height
  xx, yy = np.meshgrid(x, y)

  background = np.zeros_like(array)
  for d in range(1, config.background_config.degree + 1):
    high = config.background_config.max_amplitude / d
    for d2 in range(d + 1):
      coeff = rng.uniform(0, high)
      background += coeff * xx ** (d - d2) * yy**d2

  result = array + background
  result = result - result.min()  # min is exactly 0

  return result


def generate_cell_frame(
  config: SampleDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
  """Draw a frame and a mask for the inside region.

  Draws a random quadrilateral frame (4 corners, each inset from the
  array's actual corners), blurs it, and combines it with the input
  array: pixels inside the frame keep the original `array` values,
  and the (blurred) frame line itself is added on top everywhere.

  Parameters
  ----------
  config
    Configuration for the data generation

  Returns
  -------
  frame
    A numpy array containing the frame of the cell.
  interior_mask
    Same shape as `array`, 1.0 for pixels inside the quadrilateral
    spanned by the 4 sampled corners and not part of the blurred frame
    line (``frame_blurred < 0.5``), 0.0 elsewhere.

  """
  rng = np.random.default_rng(config.seed)
  H, W = config.cell_size

  def sample_offset() -> int:
    return int(rng.integers(7, 11))

  # Inset corners: (x, y) with x = column, y = row
  top_left = (sample_offset(), sample_offset())
  top_right = (W - 1 - sample_offset(), sample_offset())
  bottom_right = (W - 1 - sample_offset(), H - 1 - sample_offset())
  bottom_left = (sample_offset(), H - 1 - sample_offset())
  corners = [top_left, top_right, bottom_right, bottom_left]

  frame = np.zeros((H, W), dtype=float)

  for p0, p1 in zip(corners, corners[1:] + corners[:1]):
    _draw_line(p0, p1, frame)

  frame_blurred = _widen_lines(frame)

  # Interior mask: which pixel centers fall inside the corner quadrilateral.
  y_idx, x_idx = np.mgrid[0:H, 0:W]
  pixel_centers = np.column_stack((x_idx.ravel(), y_idx.ravel()))
  polygon_path = Path(corners)
  inside = polygon_path.contains_points(pixel_centers).reshape(H, W)
  # Account for the widened frame
  interior_mask = np.logical_and(inside, frame_blurred < 0.5).astype(float)

  return frame_blurred, interior_mask


def _create_edge_nodes(config: SampleDataConfig) -> list[tuple[int, int]]:
  """Create `n_nodes` random, distinct node coordinates lying on the edges (perimeter) of a sample grid.

  Parameters
  ----------
  config
    Configuration for the data generation

  Returns
  -------
  list of tuple of int
    `n_nodes` distinct integer coordinates on the grid's perimeter, as
    (x, y) pairs (i.e. x in {0, width-1} or y in {0, height-1}),
    ordered clockwise around the perimeter starting at the top-left.

  Raises
  ------
  ValueError
    If `n_nodes` exceeds the number of distinct perimeter pixels
    available (``2 * height + 2 * width - 4``) for the given
    `sample_grid`.

  """
  assert config.structure_config is not None

  height, width = config.structure_config.structure_grid
  rng = np.random.default_rng(config.seed)

  edge_coords = set()
  for x in range(width):
    edge_coords.add((x, 0))
    edge_coords.add((x, height - 1))
  for y in range(height):
    edge_coords.add((0, y))
    edge_coords.add((width - 1, y))
  edge_coords = list(edge_coords)

  n_structures = config.structure_config.n_structures
  if n_structures > len(edge_coords):
    raise ValueError(
      f"n_nodes={n_structures} exceeds the number of distinct perimeter "
      f"pixels ({len(edge_coords)}) for a grid of shape {config.structure_config.structure_grid}."
    )

  chosen_idx = rng.choice(len(edge_coords), size=n_structures, replace=False)
  chosen = [edge_coords[i] for i in chosen_idx]

  def perimeter_position(p):
    x, y = p
    if y == 0:
      return x
    elif x == width - 1:
      return width + y
    elif y == height - 1:
      return width + height + (width - 1 - x)
    else:
      return 2 * width + height + (height - 1 - y)

  chosen.sort(key=perimeter_position)
  return chosen


def _connect_nodes(nodes: list[tuple[int, int]], config: SampleDataConfig):
  """Connect edge nodes into a fully-linked network with orthogonal wiring.

  Repeatedly picks the next unconnected node and links it to its
  (Manhattan-)nearest eligible partner. When two nodes cannot be joined
  by a single straight (horizontal/vertical) segment, one or two new
  "elbow" nodes are inserted (and appended to `nodes`) so that every
  final connection is a straight horizontal or vertical line -- ready to
  be drawn with `_draw_line`.

  Parameters
  ----------
  nodes : list of tuple of int
    Initial node coordinates, as (x, y) pairs (e.g. the output of
    `_create_edge_nodes`). This list is mutated in place: new elbow
    nodes created during connection are appended directly onto it.
  config
    Configuration for the data generation

  Returns
  -------
  nodes : list of tuple of int
    The (mutated) input node list, extended with any newly-inserted
    elbow nodes.
  node_connection_map : ndarray of int
    Array the same length as the returned `nodes`, where
    ``node_connection_map[i]`` is the index (into `nodes`) that node
    `i` is connected to.

  """
  assert config.structure_config is not None
  rng = np.random.default_rng(config.seed)
  structure_grid = config.structure_config.structure_grid

  def dist(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

  node_connection_map = np.array([-1] * len(nodes))
  while np.any(node_connection_map == -1):
    index_next = np.where(node_connection_map == -1)[0][0]
    next_node = nodes[index_next]

    index_partner = -1
    min_distance = np.inf
    for _index_partner, node in enumerate(nodes):
      if _index_partner == index_next:
        continue
      if (node[0] == 0 or node[0] == (structure_grid[0] - 1)) and node[0] == next_node[0]:
        continue
      if (node[1] == 0 or node[1] == (structure_grid[1] - 1)) and node[1] == next_node[1]:
        continue

      current_dist = dist(node, nodes[index_next])
      if current_dist < min_distance:
        min_distance = current_dist
        index_partner = _index_partner

    partner_node = nodes[index_partner]
    # Nodes are on a line
    if partner_node[0] == next_node[0] or partner_node[1] == next_node[1]:
      node_connection_map[index_next] = index_partner
      node_connection_map[index_partner] = index_next
    # Nodes are on touching edges
    elif partner_node[0] in [0, structure_grid[0] - 1] and next_node[1] in [0, structure_grid[1] - 1]:
      nodes.append((next_node[0], partner_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index
      node_connection_map[index_partner] = new_node_index
      node_connection_map = np.append(node_connection_map, index_partner)
    elif partner_node[1] in [0, structure_grid[1] - 1] and next_node[0] in [0, structure_grid[0] - 1]:
      nodes.append((partner_node[0], next_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index
      node_connection_map[index_partner] = new_node_index
      node_connection_map = np.append(node_connection_map, index_partner)
    # Nodes are on opposite edges
    elif partner_node[0] in [0, structure_grid[0] - 1] and next_node[0] in [0, structure_grid[0] - 1]:
      random_height = int(rng.uniform(1, structure_grid[0]))

      nodes.append((random_height, next_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index

      nodes.append((random_height, partner_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_partner] = new_node_index

      node_connection_map = np.append(node_connection_map, new_node_index)
      node_connection_map = np.append(node_connection_map, index_partner)
    elif partner_node[1] in [0, structure_grid[1] - 1] and next_node[1] in [0, structure_grid[1] - 1]:
      random_width = int(rng.uniform(1, structure_grid[1]))

      nodes.append((next_node[0], random_width))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index

      nodes.append((partner_node[1], random_width))
      new_node_index = len(nodes) - 1
      node_connection_map[index_partner] = new_node_index

      node_connection_map = np.append(node_connection_map, new_node_index)
      node_connection_map = np.append(node_connection_map, index_partner)
    # One node is in the center of the field
    elif next_node[0] in [0, structure_grid[0] - 1]:
      nodes.append((partner_node[0], next_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index
      node_connection_map[index_partner] = new_node_index
      node_connection_map = np.append(node_connection_map, index_partner)
    else:
      nodes.append((next_node[0], partner_node[1]))
      new_node_index = len(nodes) - 1
      node_connection_map[index_next] = new_node_index
      node_connection_map[index_partner] = new_node_index
      node_connection_map = np.append(node_connection_map, index_partner)

  return nodes, node_connection_map


def _upscale(nodes: list[tuple[int, int]], config: SampleDataConfig) -> list[tuple[int, int]]:
  """Scale nodes from sample grid to target grid.

  Rescale node coordinates from a smaller sample grid onto a larger
  target grid, using "align corners" scaling: a node sitting on an edge
  (or corner) of the sample grid lands exactly on the corresponding
  edge (or corner) of the target grid.

  Parameters
  ----------
  target_grid : tuple of int
    Shape of the grid to upscale onto, as (height, width).
  sample_grid : tuple of int
    Shape of the grid the nodes currently live on, as (height, width).
  nodes : list of tuple of int
    Node coordinates on the sample grid, as (x, y) pairs.
  config
    Configuration for the data generation

  Returns
  -------
  list of tuple of int
    Node coordinates rescaled onto the target grid, as (x, y) pairs.
    Coordinates are truncated to int.

  """
  assert config.structure_config is not None
  target_h, target_w = config.frame_size
  sample_h, sample_w = config.structure_config.structure_grid

  scale_x = (target_w - 1) / (sample_w - 1) if sample_w > 1 else 0.0
  scale_y = (target_h - 1) / (sample_h - 1) if sample_h > 1 else 0.0

  return [(int(x * scale_x), int(y * scale_y)) for x, y in nodes]


def generate_structures(config: SampleDataConfig) -> np.ndarray:
  """Draw structures of connected line segments onto a blank array.

  Parameters
  ----------
  config
    Configuration for the data generation

  Returns
  -------
  ndarray
    Array of shape `frame_size` containing the line structures.

  """
  edge_nodes = _create_edge_nodes(config)
  sample_nodes, connections = _connect_nodes(edge_nodes, config)
  nodes = _upscale(sample_nodes, config)

  structures = np.zeros(config.frame_size, dtype=float)
  for i, j in enumerate(connections):
    _draw_line(nodes[i], nodes[j], structures)
  return _widen_lines(structures)


def _fade(t: np.ndarray) -> np.ndarray:
  """Perlin's quintic fade curve (smootherstep), 6t^5 - 15t^4 + 10t^3."""
  return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_noise_2d(shape: tuple[int, int], period: float, seed: int | None = None) -> np.ndarray:
  """Generate 2D Perlin noise on a grid of the given shape.

  Parameters
  ----------
  shape : tuple[int, int]
    (height, width) of the output noise array.
  period : float
    Lattice spacing in pixels: the distance between independent random
    gradient vectors. Larger values produce smoother, lower-frequency
    noise; smaller values produce busier, higher-frequency noise.
  seed : int or None, optional
    Seed for the random number generator. Default is None
    (non-deterministic).

  Returns
  -------
  np.ndarray
    Array of shape `shape` containing Perlin noise, roughly in the
    range [-1, 1].

  """
  rng = np.random.default_rng(seed)
  height, width = shape

  # Random unit gradient vectors at each lattice point.
  n_cells_y = int(np.ceil(height / period)) + 1
  n_cells_x = int(np.ceil(width / period)) + 1
  angles = 2 * np.pi * rng.random((n_cells_y, n_cells_x))
  gradients = np.dstack((np.cos(angles), np.sin(angles)))  # (n_cells_y, n_cells_x, 2)

  # Pixel coordinates expressed in lattice-cell units.
  y = np.arange(height) / period
  x = np.arange(width) / period
  yy, xx = np.meshgrid(y, x, indexing="ij")

  y0 = np.floor(yy).astype(int)
  x0 = np.floor(xx).astype(int)
  y1 = y0 + 1
  x1 = x0 + 1

  fy = yy - y0
  fx = xx - x0

  def dot_grid_gradient(iy: np.ndarray, ix: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    g = gradients[iy, ix]
    return g[..., 0] * dx + g[..., 1] * dy

  n00 = dot_grid_gradient(y0, x0, fx, fy)
  n10 = dot_grid_gradient(y0, x1, fx - 1, fy)
  n01 = dot_grid_gradient(y1, x0, fx, fy - 1)
  n11 = dot_grid_gradient(y1, x1, fx - 1, fy - 1)

  u = _fade(fx)
  v = _fade(fy)

  nx0 = n00 * (1 - u) + n10 * u
  nx1 = n01 * (1 - u) + n11 * u
  value = nx0 * (1 - v) + nx1 * v

  # Theoretical max magnitude of the interpolated dot products is
  # sqrt(2)/2; rescale so the output roughly spans [-1, 1].
  return value / (np.sqrt(2) / 2)


def generate_noise(config: SampleDataConfig) -> np.ndarray:
  """Generate a 2D array of Perlin noise according to a sample data config.

  Parameters
  ----------
  config : SampleDataConfig
    Provides `frame_size` (output shape) and `noise_config` (amplitude
    and noise_level for the Perlin noise).

  Returns
  -------
  np.ndarray
    Array of shape `config.frame_size` containing Perlin noise scaled
    by `config.noise_config.amplitude`.

  """
  assert config.noise_config is not None
  height, width = config.frame_size
  period = min(height, width) / config.noise_config.noise_level
  noise = _perlin_noise_2d(config.frame_size, period, seed=config.seed)
  return config.noise_config.amplitude * (noise - np.min(noise))


def generate_sample_frame(config: SampleDataConfig) -> np.ndarray:
  """Generate a single random frame with a cell.

  Parameters
  ----------
  config
    Configuration for the data generation

  Returns
  -------
  The resulting image frame in form of a 2d numpy array.

  """
  frame = np.zeros(config.frame_size)

  cell = generate_gaussian_field(config)
  if config.background_config is not None:
    cell = add_background(cell, config)

  cell_frame, inside = generate_cell_frame(config)

  if config.structure_config is not None:
    frame += generate_structures(config)

  # Embed the cell
  rng = np.random.default_rng(config.seed)
  embedding_offset = (
    int(rng.integers(0, config.frame_size[0] - config.cell_size[0])),
    int(rng.integers(0, config.frame_size[1] - config.cell_size[1])),
  )
  bottom, top = embedding_offset[0], embedding_offset[0] + config.cell_size[0]
  left, right = embedding_offset[1], embedding_offset[1] + config.cell_size[1]
  frame[bottom:top, left:right] *= 1 - inside
  frame[bottom:top, left:right] += cell_frame
  frame[frame > 1] = 1.0  # Renormalize
  frame[bottom:top, left:right] += cell * inside

  if config.noise_config is not None:
    frame += generate_noise(config)
  return frame
