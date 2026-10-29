import numpy as np
import warnings

# Suppress DeprecationWarnings (will need to fix one day)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- CONFIGURATION ---
ACTIVE_LAYER = 2 # Change to 0, 1, or 2 to switch between layers

layers = {
    #input layer: 16->8
    0: {
        'inputs': [0x05, 0x0f, 0x0d, 0x00, 0x24, 0xe9, 0xea, 0x04, 0x14, 0xf9, 0xf0, 0xfb, 0xfd, 0xeb, 0x0f, 0xef],
        'weights': [0xe2, 0x04, 0xf6, 0x08, 0xfd, 0x00, 0x0e, 0xfc, 0xf8, 0xf7, 0x02, 0x0b, 0x04, 0x05, 0xf3, 0x08], # PE 0
        'bias': 0x67,
        'use_relu': True,
        'input_bits': 8,
        'shift': 0
    },
    #layer 1: 8->4
    1: {
        'inputs': [0x0, 0x0, 0x9b5, 0x40f, 0x349, 0x494, 0x0, 0x0],
        'weights': [0xc7, 0xe0, 0xf6, 0x04, 0x00, 0x07, 0x0c, 0x26], # PE 0
        'bias': 0x22d,
        'use_relu': False,
        'input_bits': 16,
        'shift': 5
    },
    #layer 2: 4->8
    2: {
        'inputs': [0xfe8b, 0x8a6, 0xf8c1, 0x1429],
        'weights': [0xc5, 0x21, 0xd6, 0xcf], # PE 0
        'bias': 0xfffffca1,
        'use_relu': True,
        'input_bits': 16,
        'shift': 5
    }
}

# --- SCRIPT ---
cfg = layers[ACTIVE_LAYER]

print(f"=== Evaluating Layer {ACTIVE_LAYER} ===")

# Weights are 8-bit in both layers
w_int32 = np.array(cfg['weights'], dtype=np.uint8).view(np.int8).astype(np.int32)

# Inputs are 8-bit in Layer 0, and 16-bit in Layer 1/2
if cfg['input_bits'] == 16:
    # Must view as signed int16 FIRST to interpret the negative bits properly, then widen to int32
    i_int32 = np.array(cfg['inputs'], dtype=np.uint16).view(np.int16).astype(np.int32)
    i_mask, i_fmt = 0xFFFF, "04x"
else:
    i_int32 = np.array(cfg['inputs'], dtype=np.uint8).view(np.int8).astype(np.int32)
    i_mask, i_fmt = 0xFF, "02x"

running_sum = 0

# Widened the padding columns to handle larger negative values
print(f"{'Idx':<3} | {'Weight (Hex)':<13} | {'Input (Hex)':<16} | {'Product (Hex)':<24} | {'Running Sum (Hex)':<24}")
print("-" * 92)

for idx, (w, i) in enumerate(zip(w_int32, i_int32)):
    product = w * i
    running_sum += product
    
    # Cast numpy scalars to standard Python ints before applying bitwise masks
    w_val = int(w)
    i_val = int(i)
    p_val = int(product)
    rs_val = int(running_sum)
    
    w_str = f"{w_val} (0x{w_val & 0xFF:02x})"
    
    # Dynamically format input hex string based on layer bit-width
    i_str = f"{i_val} (0x{i_val & i_mask:{i_fmt}})"
    
    # Using 32-bit masks (0xFFFFFFFF) because the products easily exceed 16 bits
    p_str = f"{p_val} (0x{p_val & 0xFFFFFFFF:08x})"
    rs_str = f"{rs_val} (0x{rs_val & 0xFFFFFFFF:08x})"
    
    # Widened formatting here to match the header
    print(f"{idx:<3} | {w_str:<13} | {i_str:<16} | {p_str:<24} | {rs_str:<24}")

print("-" * 92)

# Force Python to treat the hex bias as a signed 32-bit integer
bias_signed = np.int32(cfg['bias']).item()
final_result = int(running_sum) + bias_signed

print(f"Final dot product (before bias): {int(running_sum)} (0x{int(running_sum) & 0xFFFFFFFF:08x})")
print(f"Final dot product (after bias 0x{cfg['bias'] & 0xFFFFFFFF:08x}):  {final_result} (0x{final_result & 0xFFFFFFFF:08x})")

# Apply shift (Truncating towards zero to match C/hardware behavior)
if cfg['shift'] > 0:
    final_result = int(final_result / (1 << cfg['shift'])) 
    print(f"Final dot product (after shift >> {cfg['shift']}):  {final_result} (0x{final_result & 0xFFFFFFFF:08x})")

if cfg['use_relu']:
    if (final_result < 0):
        final_result = 0
    print(f"Final dot product (after ReLU):  {final_result} (0x{final_result & 0xFFFFFFFF:08x})")
