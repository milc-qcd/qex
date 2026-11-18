"""Independent MILC6/Z4 and free-HISQ Fourier/RK3 reference for random-wall tests."""
import numpy as np


class Milc6:
    def __init__(self, seed, stream):
        self.r = []
        for _ in range(7):
            seed = ((69607 + 8 * stream) * seed + 12345) & 0xFFFFFFFF
            self.r.append((seed >> 8) & 0xFFFFFF)
        self.state = ((69607 + 8 * stream) * seed + 12345) & 0xFFFFFFFF
        self.multiplier = (100005 + 8 * stream) & 0xFFFFFFFF

    def integer(self):
        r = self.r
        value = (((r[5] >> 7) | (r[6] << 17))
                 ^ ((r[4] >> 1) | (r[5] << 23))) & 0xFFFFFF
        self.r = [value] + r[:6]
        self.state = (self.state * self.multiplier + 12345) & 0xFFFFFFFF
        return value ^ ((self.state >> 8) & 0xFFFFFF)


def wall(seed, shape, source_time=None):
    nt, nz, ny, nx = shape
    volume = int(np.prod(shape))
    rng = Milc6(seed, volume)
    limit = (1 << 24) - ((1 << 24) % nt)
    value = rng.integer()
    while value >= limit:
        value = rng.integer()
    t0 = value % nt if source_time is None else source_time
    source = np.zeros((*shape, 3), dtype=complex)
    phases = (1, 1j, -1, -1j)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index = x + nx * (y + ny * (z + nz * t0))
                rng = Milc6(seed, index)
                source[t0, z, y, x] = [
                    phases[rng.integer() >> 22] for _ in range(3)]
    return t0, source


def shift(field, axis, hop):
    result = np.roll(field, -hop, axis=axis).copy()
    if axis == 0:
        # Quark fields are antiperiodic in Euclidean time.
        sign = (-1.0) ** np.floor_divide(np.arange(field.shape[0]) + hop,
                                        field.shape[0])
        result *= sign[:, None, None, None, None]
    return result


def derivative(field):
    t, z, y, x = np.indices(field.shape[:-1])
    result = np.zeros_like(field)
    for axis, parity in [(3, t), (2, x + t), (1, x + y + t),
                         (0, np.zeros_like(t))]:
        eta = ((-1.0) ** parity)[..., None]
        result += eta * (
            (9 / 8) * (shift(field, axis, 1) - shift(field, axis, -1))
            - (1 / 24) * (shift(field, axis, 3) - shift(field, axis, -3))) / 2
    return result


def reference(seed, geometry, mass, flow_steps=(), source_time=None):
    shape = tuple(reversed(geometry))  # t,z,y,x; color is the last axis
    t0, source = wall(seed, shape, source_time=source_time)
    nt = shape[0]
    twist = np.exp(1j * np.pi * np.arange(nt) / nt)[:, None, None, None, None]
    denominator = np.full(shape, mass * mass)
    for axis, extent in enumerate(shape):
        momentum = 2 * np.pi * np.fft.fftfreq(extent)
        if axis == 0:
            momentum += np.pi / extent
        derivative_eigenvalue = (9 / 8) * np.sin(momentum) - np.sin(3 * momentum) / 24
        broadcast_shape = [1] * 4
        broadcast_shape[axis] = extent
        denominator += derivative_eigenvalue.reshape(broadcast_shape) ** 2
    normal_solution = twist * np.fft.ifftn(
        np.fft.fftn(source / twist, axes=(0, 1, 2, 3))
        / denominator[..., None], axes=(0, 1, 2, 3))
    prop = mass * normal_solution - derivative(normal_solution)
    residual = mass * prop + derivative(prop) - source
    relative_residual2 = np.vdot(residual, residual).real / np.vdot(source, source).real
    assert relative_residual2 < 1e-25, relative_residual2

    # In the free field the directional HISQ flow is diagonal in momentum.
    # Its three-stage RK step has R(z)=1+z+z^2/2+z^3/6, z=-h*lambda.
    if flow_steps:
        multiplier = np.ones(shape)
        eigenvalue = denominator - mass * mass
        for step in flow_steps:
            z = -step * eigenvalue
            multiplier *= 1 + z + z*z/2 + z*z*z/6
        prop = twist * np.fft.ifftn(
            np.fft.fftn(prop / twist, axes=(0, 1, 2, 3))
            * multiplier[..., None], axes=(0, 1, 2, 3))

    spatial_volume = np.prod(shape[1:])
    pp = np.sum(abs(prop) ** 2, axis=(1, 2, 3, 4)) / spatial_volume
    c1 = np.sum(prop.conj() * (9 / 8) * shift(prop, 0, 1),
                axis=(1, 2, 3, 4))
    c3 = np.sum(prop.conj() * (-1 / 24) * shift(prop, 0, 3),
                axis=(1, 2, 3, 4))
    current = -(c1 + c3 + np.roll(c3, 1) + np.roll(c3, 2)) / spatial_volume
    thin = -np.sum(prop.conj() * shift(prop, 0, 1),
                   axis=(1, 2, 3, 4)) / spatial_volume
    return t0, np.roll(pp, -t0), np.roll(current, -t0), np.roll(thin, -t0)
