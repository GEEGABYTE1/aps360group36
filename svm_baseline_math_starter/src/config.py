from dataclasses import dataclass

@dataclass
class Config:
    crop_size: int = 48
    min_area: int = 20
    hog_pixels_per_cell: tuple = (8, 8)
    hog_cells_per_block: tuple = (2, 2)
    hog_orientations: int = 9
    zoning_rows: int = 6
    zoning_cols: int = 6
    # reconstruction thresholds (relative to base box height/width)
    tau_sup_y: float = 0.25
    tau_sub_y: float = 0.25
    tau_x_right: float = 0.1
