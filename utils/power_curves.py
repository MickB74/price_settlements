import numpy as np
import pandas as pd

def get_normalized_power(wind_speed_series, turbine_type="GENERIC"):
    """
    Applies a specific power curve to a wind speed series (m/s).
    Returns a normalized series (0.0 to 1.0).
    """
    if turbine_type == "VESTAS_V163":
        return _curve_vestas_v163(wind_speed_series)
    elif "GE" in turbine_type and "2.8" in turbine_type:
        return _curve_ge_2x(wind_speed_series) # 2.82 is similar to 2.5 curve shape
    elif "V110" in turbine_type:
         return _curve_generic_iec2(wind_speed_series) # V110 is standard IEC II
    elif "V136" in turbine_type:
         return _curve_vestas_v163(wind_speed_series) # V136 is also low wind/high efficiency, similar to V163 shape
    elif turbine_type == "GE_2X":
        return _curve_ge_2x(wind_speed_series)
    elif turbine_type == "GE_3X":
        return _curve_ge_3x(wind_speed_series)
    elif turbine_type == "NORDEX_N163":
        return _curve_nordex_n163(wind_speed_series)
    elif turbine_type == "NORDEX_N149":
        return _curve_nordex_n149(wind_speed_series)
    else:
        return _curve_generic_iec2(wind_speed_series)

def get_curve_for_specs(manuf, model, rotor_m=None):
    """
    Factory to guess the best curve based on Metadata.
    """
    model = str(model).upper()
    manuf = str(manuf).upper()
    
    if "GE" in manuf or "GE" in model:
        if "2.8" in model or "127" in model:
            return "GE_2X"
        if "3.6" in model or "154" in model:
            return "GE_3X"
            
    if "VESTAS" in manuf or "VESTAS" in model:
        if "V163" in model or "V150" in model:
            return "VESTAS_V163"
        if "V136" in model:
            return "VESTAS_V163" # Close enough proxy
        if "V110" in model:
            return "GENERIC" # V110-2.0 is a workhorse, standard curve
            
    if "NORDEX" in manuf:
        if "163" in model or "5." in model:
            return "NORDEX_N163"
            
    # New Mappings from Enrichment
    if "ACCIONA" in manuf or "AW125" in model or "AW116" in model:
        return "GE_2X" # Proxy for 3MW class / 125m rotor
        
    if "V117" in model or "V126" in model:
        return "GE_2X" # Proxy for 3.3MW class
            
    # Heuristic based on Rotor Diameter
    # >130m usually means Low Wind / High Efficiency curve
    if rotor_m and rotor_m > 130:
        return "VESTAS_V163"
        
    return "GENERIC"

def _curve_generic_iec2(v):
    """
    Standard Cubic Power Curve (IEC Class 2 proxy).
    Cut-in: 3 m/s
    Rated: 12 m/s
    Cut-out: 25 m/s
    """
    # Vectorized implementation for speed
    power = np.zeros_like(v)
    
    # Cubic region (3 to 12)
    mask_cubic = (v >= 3.0) & (v < 12.0)
    power[mask_cubic] = ((v[mask_cubic] - 3.0) / 9.0) ** 3
    
    # Rated region (12 to 25)
    mask_rated = (v >= 12.0) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power

def _curve_vestas_v163(v):
    """
    Vestas V163-4.5 MW (Low Wind Specialist).
    High capacity factor machine.
    Cut-in: 3.0 m/s
    Rated: ~10.5 m/s (Reaches full power earlier)
    Cut-out: 25.0 m/s
    """
    power = np.zeros_like(v)
    
    # Cubic region (3 to 10.5) - Steeper ramp
    # ((v - 3) / (10.5 - 3)) ** 2.8 (Slightly less than cubic to match modern control)
    mask_ramp = (v >= 3.0) & (v < 10.5)
    power[mask_ramp] = ((v[mask_ramp] - 3.0) / 7.5) ** 2.5 
    
    # Rated region (10.5 to 25)
    mask_rated = (v >= 10.5) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power

def _curve_ge_2x(v):
    """
    GE 2.5-127 (Workhorse).
    Cut-in: 3.0 m/s
    Rated: 11.0 m/s
    Cut-out: 25.0 m/s
    """
    power = np.zeros_like(v)
    
    mask_ramp = (v >= 3.0) & (v < 11.0)
    power[mask_ramp] = ((v[mask_ramp] - 3.0) / 8.0) ** 3
    
    
    mask_rated = (v >= 11.0) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power

def _curve_ge_3x(v):
    """
    GE Vernova 3.6-154 (Modern Mainstream).
    Designed for medium wind speeds but large rotor gives good low-end.
    Cut-in: 3.0 m/s
    Rated: 10.5 m/s
    Cut-out: 25.0 m/s
    """
    power = np.zeros_like(v)
    
    # Efficient ramp due to large rotor
    mask_ramp = (v >= 3.0) & (v < 10.5)
    power[mask_ramp] = ((v[mask_ramp] - 3.0) / 7.5) ** 2.6
    
    mask_rated = (v >= 10.5) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power

def _curve_nordex_n163(v):
    """
    Nordex N163/5.X (Low Wind / Large Rotor).
    Very aggressive low wind performance.
    Cut-in: 3.0 m/s
    Rated: 10.0 m/s (Reaches rated very early)
    Cut-out: 25.0 m/s
    """
    power = np.zeros_like(v)
    
    # Aggressive ramp
    mask_ramp = (v >= 3.0) & (v < 10.0)
    power[mask_ramp] = ((v[mask_ramp] - 3.0) / 7.0) ** 2.5
    
    mask_rated = (v >= 10.0) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power

def _curve_nordex_n149(v):
    """
    Nordex N149/4.X-5.X.
    Standard-to-Low wind machine, slightly higher specific power than N163.
    Cut-in: 3.0 m/s
    Rated: ~11.5 m/s
    Cut-out: 25.0 m/s
    """
    power = np.zeros_like(v)
    
    # Cubic region
    # ((v - 3) / (11.5 - 3)) ** 2.8
    mask_ramp = (v >= 3.0) & (v < 11.5)
    power[mask_ramp] = ((v[mask_ramp] - 3.0) / 8.5) ** 2.8
    
    mask_rated = (v >= 11.5) & (v < 25.0)
    power[mask_rated] = 1.0
    
    return power
    

