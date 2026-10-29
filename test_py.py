import numpy as np

#these are for layer 0
#inputs = [0x05, 0x0f, 0x0d, 0x00, 0x24, 0xe9, 0xea, 0x04, 0x14, 0xf9, 0xf0, 0xfb, 0xfd, 0xeb, 0x0f, 0xef]
#ReLU = True
#these are for pe 0, layer 0
#weights = [0xe2, 0x04, 0xf6, 0x08, 0xfd, 0x00, 0x0e, 0xfc, 0xf8, 0xf7, 0x02, 0x0b, 0x04, 0x05, 0xf3, 0x08]
#bias = 0x67
#these are for pe 1, layer 0
#weights = [0x0d, 0xf6, 0x08, 0xfd, 0xfd, 0x11, 0x04, 0xf7, 0x06, 0xf4, 0xfb, 0xff, 0x25, 0x09, 0xfb, 0xfc, 0xed]
#bias = 0xb0

# these are for layer 1
inputs = [0x0, 0x0, 0x9b5, 0x40f, 0x349, 0x494, 0x0, 0x0]
ReLu = False
#these are for pe 0, layer 1
weights = [0xc7, 0xe0, 0xf6, 0x04, 0x00, 0x07, 0x0c, 0x26]
bias = 0x22d

#expected = [0x01, 0x0c, 0xfc, 0x00, 0x16, 0xed, 0xe7, 0x10, 0x0f, 0xfc, 0xf7, 0xfe, 0xfe, 0xfd, 0x0c, 0xec]

# Load and cast to 16-bit signed integers
w_int16 = np.array(weights, dtype=np.uint8).view(np.int8).astype(np.int16)
#these are for layer 0
#i_int16 = np.array(inputs, dtype=np.uint8).view(np.int8).astype(np.int16)
#these are for layer 1
i_int16 = np.array(inputs, dtype=np.uint16).view(np.int16).astype(np.int16) #may not be right


running_sum = 0
ReLU = True

print(f"{'Idx':<3} | {'Weight (Hex)':<13} | {'Input (Hex)':<13} | {'Product (Hex)':<16} | {'Running Sum (Hex)':<17}")
print("-" * 73)

for idx, (w, i) in enumerate(zip(w_int16, i_int16)):
    product = w * i
    running_sum += product
    
    # Cast numpy scalars to standard Python ints before applying bitwise masks
    w_val = int(w)
    i_val = int(i)
    p_val = int(product)
    rs_val = int(running_sum)
    
    w_str = f"{w_val} (0x{w_val & 0xFF:02x})"
    i_str = f"{i_val} (0x{i_val & 0xFF:02x})"
    p_str = f"{p_val} (0x{p_val & 0xFFFF:04x})"
    rs_str = f"{rs_val} (0x{rs_val & 0xFFFF:04x})"
    
    print(f"{idx:<3} | {w_str:<13} | {i_str:<13} | {p_str:<16} | {rs_str:<17}")

print("-" * 73)

# Add bias
final_result = int(running_sum) + bias


print(f"Final dot product (before bias): {int(running_sum)} (0x{int(running_sum) & 0xFFFF:04x})")
print(f"Final dot product (after bias {hex(bias)}):  {final_result} (0x{final_result & 0xFFFF:04x})")
if ReLU:
    if (final_result < 0):
        final_result = 0
    print(f"Final dot product (after ReLU):  {final_result} (0x{final_result & 0xFFFF:04x})")
