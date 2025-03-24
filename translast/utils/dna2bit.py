_base_to_bit = {
        'T': '00', 't': '00',
        'C': '01', 'c': '01',
        'A': '10', 'a': '10',
        'G': '11', 'g': '11',
        'N': '00'} # Like T in two bits

_bits_to_base = {
        0: 'T',
        1: 'C',
        2: 'A',
        3: 'G',
        4: 'N'} # Like T in two bits

_bits_to_comp = {
        0: 'C',
        1: 'T',
        2: 'G',
        3: 'A',
        4: 'N'}

def byte_to_bases(i):
    """convert one byte to the four bases it encodes"""
    c = (i >> 4) & 0xf
    f = i & 0xf
    cc = (c >> 2) & 0x3
    cf = c & 0x3
    fc = (f >> 2) & 0x3
    ff = f & 0x3
    return map(lambda b: _bits_to_base[b], (cc, cf, fc, ff))

def byte_to_comp(i):
    """convert one byte to the four bases it encodes"""
    c = (i >> 4) & 0xf
    f = i & 0xf
    cc = (c >> 2) & 0x3
    cf = c & 0x3
    fc = (f >> 2) & 0x3
    ff = f & 0x3
    return map(lambda b: _bits_to_comp[b], (cc, cf, fc, ff))

def bases_to_byte(bs):
    return int(''.join([_base_to_bit[n] for n in bs]), 2)

def decode_n_blocks(byts):
    blockCount = int.from_bytes(byts[:4], byteorder='big')
    blockStarts = []
    blockSizes = []
    for k in range(4, 4*(blockCount+1), 4):
        blockStarts.append(int.from_bytes(byts[k:k + 4], byteorder='big'))

    for k in range(4*(blockCount+1), 4*(2*blockCount+1), 4):
        blockSizes.append(int.from_bytes(byts[k:k + 4], byteorder='big'))
        
    return blockCount, blockStarts, blockSizes

def decode_dna(byts):
    return [n for byt in byts for n in byte_to_bases(byt)]

def rev_comp(byt):
    res = ~byt
    res = ((res >> 2 & 0x33) | (res & 0x33) << 2)
    res = ((res >> 4 & 0x0F) | (res & 0x0F) << 4)
    return res

def decode_rco(byts):
    return [n for byt in byts[::-1] for n in byte_to_comp(rev_comp(byt))]

def bytes2dna(byts):
    byts = byts.numpy()
    dnaSize = int.from_bytes(byts[:4], byteorder='big')
    # Unpack blocks of Ns in the file
    blockCount, blockStarts, blockSizes = decode_n_blocks(byts[4:])
    # DNA unpacked to nucleotide list
    dnalist = decode_dna(byts[8*(blockCount+1):])[:dnaSize]
    # Change unknown nucleotide in sequence
    for st, bs in zip(blockStarts, blockSizes):
        dnalist[st:st+bs] = bs*['N']
    return [dnalist]
