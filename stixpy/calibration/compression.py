"""
STIX integer compression is a quasi-exponential 1 byte compression of integer data, both positive
or negative depending on the parameterization.  For the decompression it returns the mid-point value
of the possible compressed values that could have given the input.

The compression algorithm is applied to an integer value, V, to yield a compressed value, C, as
follows:

1. If  V < 0, set s=1 and set V = -V
2. If  V < 2^(M+1), set C = V and skip to step 6
3. If  V >= 2^(M+1), shift V right until 1s (if any) appear only in LS M+1 bits
4. Exponent, e = number of shifts+ 1
5. Mantissa, m = LS M bits of the shifted value.
6. Set C = m + 2^M * e
7. If S=1, set msb of C = s

The algorithm family has the following properties:

* The compressed value is exact for input absolute values up to 2^(M+1)
* RMS fractional error is in range: 1./(sqrt(12)*2^M) to 1/(sqrt(1/12)*s^(M+1))
* The maximum absolute value that can be compressed is given by: 2^(2^K-2) x (2^(M+1)-1)
  The number of bits required to hold the compressed value is S+K+M

References
----------
STIX-TN-0117-FHNW
"""
import numpy as np

__all__ = ['compress', 'decompress', 'CompressionRangeError', 'CompressionSchemeParameterError',
           'NonIntegerCompressionError']


def compress(values, *, s, k, m):
    """
    Compress values according to parameters.

    Parameters
    ----------
    values : array-like (int)
        Values to be compressed
    s : int
        Number of sign bits, 0 for positive only values 1 for signed
    k : int
        Number of bits for exponent
    m : int
        Number of bit for mantissa

    Returns
    -------
    array
        The compressed values

    Examples
    --------
    >>> comp = compress(12345, s=0, k=5, m=3)
    >>> int(comp)
    92
    >>> comp = compress(-1984, s=1, k=3, m=4)
    >>> int(comp)
    255
    """
    values = np.atleast_1d(np.array(values))
    if not np.issubdtype(values.dtype, np.integer):
        raise NonIntegerCompressionError(f'Input must be an integer type not {values.dtype}')

    total = s + k + m
    if total > 8:
        raise CompressionSchemeParameterError(f'Invalid s={s}, k={k}, m={m} '
                                              f'must sum to less than 8 not {total}')

    max_value = 2**(2**k-2)*(2**(m + 1)) - 1
    abs_range = [0, max_value] if s == 0 else [-1 * max_value, 1 * max_value]

    if values.min() < abs_range[0] or values.max() > abs_range[1]:
        raise CompressionRangeError(f'Valid input range exceeded for s={s}, k={k}, m={m} '
                                    f'input (min/max) {values.min()}/{values.max()} '
                                    f'exceeds {abs_range}')

    negative_indices = np.nonzero(values < 0)
    out = np.abs(values).astype(np.uint64)
    compress_indices = np.nonzero(out >= 2 ** (m + 1))

    if values[compress_indices].size != 0:
        kv = np.floor((np.log(np.abs(values[compress_indices])) / np.log(2.0) - m)).astype(np.int64)
        mv = (np.abs(values[compress_indices], dtype=np.int64) // 2**kv) & (2**m - 1)
        out[compress_indices] = kv + 1 << m | mv

    if s != 0 and values[negative_indices].size >= 1:
        out[negative_indices] += 128
    return out


def decompress(values, *, s, k, m, return_variance=False):
    """
    Compress values according to parameters.

    Parameters
    ----------
    values : array-like (int)
        Values to decompressed
    s : int (0, 1)
        Number of sign bits
    k : int
        Number of bits for exponent
    m : int
        Number of bits for mantissa
    return_variance : boolean (optional)

    Returns
    -------
    array
        The decompressed values

    Examples
    --------
    >>> decomp = decompress(92, s=0, k=5, m=3)
    >>> int(decomp)
    12799
    >>> decomp = decompress(255, s=1, k=3, m=4)
    >>> int(decomp)
    -2015
    """
    values = np.atleast_1d(np.array(values))
    if not np.issubdtype(values.dtype, np.integer):
        raise NonIntegerCompressionError(f'Input must be an integer type not {values.dtype}')

    values = values.astype(np.uint64)

    total = s + k + m
    if total > 8:
        raise CompressionSchemeParameterError(f'Invalid s={s}, k={k}, m={m} '
                                              f'must sum to less than 8 not {total}')

    if values.min() < 0 or values.max() > 255:
        raise CompressionRangeError(f'Compressed values must be in the range 0 to 255')

    max_value = 2**(2**k-2)*(2**(m + 1)) - 1
    uint64_info = np.iinfo(np.uint64)

    if max_value > uint64_info.max:
        raise CompressionRangeError('Decompressed value too large to fit into uint64')

    abs_values = values if s == 0 else values & 127
    out = abs_values[:].astype(np.uint64)
    negative_indices = np.nonzero(values >= 128) if s != 0 else ([])
    decompress_indices = np.nonzero(abs_values >= 2**(m+1))

    if return_variance:
        variance = np.zeros_like(values, dtype=np.float64)

    if abs_values[decompress_indices].size >= 1:
        kv = (abs_values[decompress_indices] >> m) - 1
        mv = 2**m + (abs_values[decompress_indices] & 2**m - 1)
        out[decompress_indices] = (mv << kv) + (1 << (kv - 1)) - 1
        if return_variance:
            variance[decompress_indices] = ((1 << kv)**2 + 2)/12

    if abs_values[negative_indices].size >= 1:
        out = out.astype(np.int64)
        out[negative_indices] = -1 * out[negative_indices]

    if return_variance:
        return out, variance
    else:
        return out


class NonIntegerCompressionError(Exception):
    """
    Exception raised when the input to the compression algorithm is not an integer.
    """


class CompressionSchemeParameterError(Exception):
    """
    Exception raised for invalid compression scheme parameters.
    """


class CompressionRangeError(Exception):
    """
    Exception raised due to invalid range in the input or output.
    """
