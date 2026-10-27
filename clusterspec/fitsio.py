"""Small, dependency-free FITS reader/writer for images and simple binary tables.

Enough for multi-extension 2D spectra and 1D outputs. If astropy is installed, ``read_fits``
uses it (more formats, e.g. compressed HDUs); otherwise the built-in reader handles
uncompressed images. Files written here open normally in astropy, DS9, Marz, etc.
"""
import numpy as np

BLOCK = 2880
_DT = {8: 'u1', 16: '>i2', 32: '>i4', 64: '>i8', -32: '>f4', -64: '>f8'}


# ---------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------
def _parse_value(s):
    s = s.strip()
    if s.startswith("'"):
        out, i = '', 1
        while i < len(s):
            if s[i] == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    out += "'"
                    i += 2
                    continue
                break
            out += s[i]
            i += 1
        return out.rstrip()
    v = s.split('/')[0].strip()
    if v == 'T':
        return True
    if v == 'F':
        return False
    try:
        return int(v)
    except ValueError:
        try:
            return float(v.replace('D', 'E'))
        except ValueError:
            return v


def _read_builtin(path):
    raw = open(path, 'rb').read()
    pos, hdus = 0, []
    while pos + BLOCK <= len(raw):
        hdr, done = {}, False
        while not done:
            blk = raw[pos:pos + BLOCK]
            pos += BLOCK
            for i in range(0, BLOCK, 80):
                card = blk[i:i + 80].decode('ascii', 'replace')
                key = card[:8].strip()
                if key == 'END':
                    done = True
                    break
                if card[8:10] == '= ':
                    hdr[key] = _parse_value(card[10:])
        naxis = int(hdr.get('NAXIS', 0))
        shape = [int(hdr[f'NAXIS{k}']) for k in range(naxis, 0, -1)]
        bitpix = int(hdr['BITPIX'])
        n = int(np.prod(shape)) if naxis else 0
        nbytes = (abs(bitpix) // 8) * int(hdr.get('GCOUNT', 1)) * (int(hdr.get('PCOUNT', 0)) + n)
        data = None
        if naxis and hdr.get('XTENSION', 'IMAGE') == 'IMAGE':
            data = np.frombuffer(raw[pos:pos + n * abs(bitpix) // 8], dtype=_DT[bitpix]).reshape(shape)
            data = data.astype(float)
            if 'BSCALE' in hdr or 'BZERO' in hdr:
                data = data * float(hdr.get('BSCALE', 1)) + float(hdr.get('BZERO', 0))
        pos += -(-nbytes // BLOCK) * BLOCK
        hdus.append((hdr, data))
        if not raw[pos:pos + 8].startswith(b'XTENSION'):
            break
    return hdus


def read_fits(path):
    """List of (header dict, data array or None) for every HDU."""
    try:
        from astropy.io import fits
    except ImportError:
        return _read_builtin(path)
    with fits.open(path, memmap=False) as hl:
        out = []
        for h in hl:
            data = None
            if h.data is not None and not hasattr(h.data, 'columns'):
                data = np.asarray(h.data, float)
            out.append(({k: h.header[k] for k in h.header if k not in ('COMMENT', 'HISTORY', '')},
                        data))
        return out


# ---------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------
def card(key, value, comment=''):
    """One 80-character header card."""
    if isinstance(value, (bool, np.bool_)):
        v = ('T' if value else 'F').rjust(20)
    elif isinstance(value, (int, np.integer)):
        v = str(int(value)).rjust(20)
    elif isinstance(value, (float, np.floating)):
        v = (repr(float(value)).upper() if abs(value) < 1e15 else f'{value:.12E}').rjust(20)
    else:
        s = str(value).replace("'", "''")
        s = s.ljust(8) if key == 'XTENSION' else s
        v = f"'{s[:68]}'".ljust(20)
    c = f'{key:<8}= {v}' + (f' / {comment}' if comment else '')
    return c[:80].ljust(80)


def _block(cards):
    h = ''.join(cards + ['END'.ljust(80)]).encode('ascii')
    return h + b' ' * ((-len(h)) % BLOCK)


def _pad(b):
    return b + b'\0' * ((-len(b)) % BLOCK)


def _image(data, keys, primary, extname=None, extend=False):
    data = np.asarray(data, dtype='>f4')
    cards = ([card('SIMPLE', True)] if primary else [card('XTENSION', 'IMAGE')])
    ndim = 0 if data.size == 0 else data.ndim
    cards += [card('BITPIX', -32), card('NAXIS', ndim)]
    cards += [card(f'NAXIS{i + 1}', n) for i, n in enumerate(data.shape[::-1][:ndim])]
    if primary and extend:
        cards.append(card('EXTEND', True))
    if not primary:
        cards += [card('PCOUNT', 0), card('GCOUNT', 1)]
    if extname:
        cards.append(card('EXTNAME', extname))
    cards += [card(k, v) for k, v in (keys or {}).items()]
    return _block(cards) + _pad(data.tobytes())


def _table(rec, extname):
    fmt = {'f8': 'D', 'f4': 'E', 'i4': 'J', 'i2': 'I', 'i8': 'K'}
    cards = [card('XTENSION', 'BINTABLE'), card('BITPIX', 8), card('NAXIS', 2),
             card('NAXIS1', rec.dtype.itemsize), card('NAXIS2', len(rec)), card('PCOUNT', 0),
             card('GCOUNT', 1), card('TFIELDS', len(rec.dtype.names))]
    for i, n in enumerate(rec.dtype.names, 1):
        dt = rec.dtype[n]
        f = f'{dt.itemsize}A' if dt.kind == 'S' else fmt[dt.str[1:]]
        cards += [card(f'TTYPE{i}', n), card(f'TFORM{i}', f)]
    cards.append(card('EXTNAME', extname))
    return _block(cards) + _pad(rec.tobytes())


def write_fits(path, hdus):
    """Write HDUs: a list of ('image', data, header_dict, extname) or
    ('table', numpy record array with big-endian / bytes fields, extname)."""
    out = b''
    for i, h in enumerate(hdus):
        if h[0] == 'image':
            out += _image(h[1], h[2], primary=(i == 0), extname=h[3] if len(h) > 3 else None,
                          extend=len(hdus) > 1)
        elif h[0] == 'table':
            if i == 0:
                out += _image(np.zeros(0), {}, primary=True, extend=True)
            out += _table(h[1], h[2])
    open(path, 'wb').write(out)


def write_spectrum(path, wave0, dwave, flux, err=None, header=None, crpix=1.0):
    """1D spectrum: flux in the primary HDU (linear wavelength WCS), error in an 'ERR'
    extension."""
    keys = dict(CRVAL1=float(wave0), CDELT1=float(dwave), CD1_1=float(dwave),
                CRPIX1=float(crpix), CTYPE1='LINEAR')
    keys.update(header or {})
    hdus = [('image', np.asarray(flux, float), keys)]          # NaN marks unusable pixels
    if err is not None:
        hdus.append(('image', np.asarray(err, float), {}, 'ERR'))
    write_fits(path, hdus)
