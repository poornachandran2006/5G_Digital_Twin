"""
5G Network Digital Twin Configuration
All simulation parameters based on 3GPP standards and real-world 5G NR values
"""

# Simulation grid
GRID_SIZE_M = 1000          # 1km x 1km grid in meters

# gNB (base station) parameters
NUM_GNB = 3
GNB_TX_POWER_DBM = 43.0     # typical macro cell transmit power
GNB_ANTENNA_GAIN_DB = 15.0  # typical antenna gain
GNB_MAX_PRB = 100           # max physical resource blocks per cell (5G NR 20MHz)
GNB_FREQUENCY_GHZ = 3.5     # 5G mid-band frequency

# UE (user equipment) parameters  
NUM_UE = 20
UE_RX_GAIN_DB = 0.0
UE_NOISE_FIGURE_DB = 7.0
UE_MIN_DEMAND_MBPS = 1.0
UE_MAX_DEMAND_MBPS = 20.0
UE_MAX_SPEED_MPS = 3.0      # pedestrian speed ~3 m/s

# Channel parameters
PATH_LOSS_EXPONENT = 3.5    # urban environment
NOISE_POWER_DBM = -104.0    # thermal noise at 20MHz bandwidth
SINR_MIN_DB = -6.0          # minimum usable SINR
SINR_MAX_DB = 30.0

# Simulation timing
TICK_DURATION_S = 1.0       # 1 second per tick
SIM_DURATION_S = 10800      # 3 hours of simulated time = 10800 ticks

# KPI thresholds
CELL_LOAD_WARNING = 0.70    # 70% PRB usage = warning
CELL_LOAD_CRITICAL = 0.90   # 90% PRB usage = congested

# ML parameters
SEQUENCE_LENGTH = 10        # look back 10 ticks
PREDICTION_HORIZON = 30     # predict 30 ticks ahead
