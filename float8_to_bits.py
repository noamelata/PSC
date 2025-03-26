import torch

def parse_bin(s):
    t = s.split('.')
    return int(t[0], 2) + int(t[1], 2) / 2.**len(t[1])

def num_to_float8(number):
    b = "{0:08b}".format(number)
    s, e, m = b[0], b[1:5], b[5:8]
    return float('nan') if (e == '1111' and m == '111') else (-1 if s == '1' else 1) * parse_bin(
        ('0.' if e == '0000' else '1.') + m) * 2 ** (int(e, 2) - (6 if e == '0000' else 7))

lut, inv_lut = {}, {}
for i in range(256):
    number = num_to_float8(i)
    number = torch.tensor([number], device=torch.device("cuda")).to(dtype=torch.float8_e4m3fn).to(torch.float).item()
    lut[i] = number
    inv_lut[number] = i

def float8_to_bits(number):
    return inv_lut[number.float().item()]

def bits_to_float8(bit):
    return lut[bit.item()]

def float8_vec_to_bits(vec):
    return torch.cat([torch.tensor([float8_to_bits(num)], dtype=torch.uint8) for num in vec.reshape(-1)]).reshape(vec.shape)

def bit_vec_to_float8(vec):
    return torch.cat([torch.tensor([bits_to_float8(bit)], dtype=torch.float32) for bit in vec.reshape(-1)]).reshape(vec.shape)


if __name__ == "__main__":
    print(lut.values())
    pass