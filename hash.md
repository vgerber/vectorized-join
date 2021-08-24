# Requirements

* Fast
* 32/64Bit
* Non secure
* low collision rate
* Reproducable result


# Fast

## FNV 
http://isthe.com/chongo/tech/comp/fnv/#FNV-1a
FNV-1a

```
hash = offset_basis
for each octet_of_data to be hashed
    hash = hash xor octet_of_data
    hash = hash * FNV_prime
return hash


FNV_prime
32 bit FNV_prime = 2^24 + 2^8 + 0x93 = 16777619
64 bit FNV_prime = 2^40 + 2^8 + 0xb3 = 1099511628211

offset_bias
32 bit offset_basis = 2166136261
64 bit offset_basis = 14695981039346656037
```

## CRC32
https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/hash-method-performance-paper.pdf

```
C = length of data <<< 19
while (more than 8 bytes of data left){
    data_chunk = next 8 bytes of data
    A = CRC (A, data_chunk)
    B = CRC (B, data_chunk <<< 31)
    C = C  data_chunk
}
Return A, CRC(B,C)
```


## Jenkins (One-At-A-Time)
```
uint32_t jenkins_one_at_a_time_hash(const uint8_t* key, size_t length) {
  size_t i = 0;
  uint32_t hash = 0;
  while (i != length) {
    hash += key[i++];
    hash += hash << 10;
    hash ^= hash >> 6;
  }
  hash += hash << 3;
  hash ^= hash >> 11;
  hash += hash << 15;
  return hash;
}
```

## Person Hashing

```
T = shuffle([0 ... 255])

algorithm pearson hashing is
    h := 0
    result := 0

    for each r in bytes of h
        h = T[(c[0] + r) % 256]
        for each c in C loop
            h := T[ h xor c ]
        end loop
        result = ((result << 8) | h)

    return result
```


# Low Collision
// sha history + alg
https://dl.gi.de/bitstream/handle/20.500.12116/33858/PARS2019_paper_1.pdf?sequence=1&isAllowed=y
