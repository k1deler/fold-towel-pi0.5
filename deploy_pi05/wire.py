"""Bounded JSON messages and lossless wrist PNGs, with no pickle transport."""
import json
import struct

MAX_HEADER = 128 * 1024
MAX_BLOB = 4 * 1024 * 1024


def exact(sock, count):
    buf = bytearray()
    while len(buf) < count:
        part = sock.recv(count-len(buf))
        if not part:
            raise ConnectionError('Model connection closed')
        buf.extend(part)
    return bytes(buf)


def send(sock, message, blobs=()):
    sizes = [len(blob) for blob in blobs]
    if len(sizes) > 2 or any(n > MAX_BLOB for n in sizes):
        raise ValueError('Image payload exceeds protocol limits')
    header = json.dumps(dict(message, _blob_sizes=sizes), allow_nan=False).encode()
    if not 0 < len(header) <= MAX_HEADER:
        raise ValueError('Message exceeds protocol limits')
    sock.sendall(struct.pack('!I', len(header))+header+b''.join(blobs))


def receive(sock):
    size, = struct.unpack('!I', exact(sock, 4))
    if not 0 < size <= MAX_HEADER:
        raise ValueError('Invalid JSON header length')
    message = json.loads(exact(sock, size))
    if not isinstance(message, dict):
        raise ValueError('Expected JSON object')
    sizes = message.pop('_blob_sizes', [])
    if not isinstance(sizes, list) or len(sizes) > 2 or any(
            type(n) is not int or not 0 <= n <= MAX_BLOB for n in sizes):
        raise ValueError('Invalid image payload lengths')
    return message, [exact(sock, n) for n in sizes]
