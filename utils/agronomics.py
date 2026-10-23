import numpy as np

def compute_saturation_vapor_pressure(temp_c: float) -> float:
    """Computes Saturation Vapor Pressure e_s(T) in kPa."""
    return 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

def compute_vpd(temp_c: float, relative_humidity: float) -> float:
    """Computes Vapor Pressure Deficit (VPD) in kPa."""
    e_s = compute_saturation_vapor_pressure(temp_c)
    e_a = e_s * (relative_humidity / 100.0)
    return e_s - e_a

def compute_lwd(rh_series: np.ndarray, precip_series: np.ndarray) -> float:
    """Computes Leaf Wetness Duration (LWD) in hours per day."""
    wet_hours = (rh_series >= 85.0) | (precip_series > 0.1)
    return float(np.sum(wet_hours))

def compute_dtr(temp_max: float, temp_min: float) -> float:
    """Computes Diurnal Temperature Range (DTR) in °C."""
    return temp_max - temp_min