from __future__ import annotations

import numpy as np


def new_impute_solvent(atoms, pairs, n, hbond_length=2.8, noise_std=0.07):
    if n == 1:
        atom_i = np.random.choice(atoms)["coords"]
        delta = np.random.normal(0, 1, 3)
        imputed_coords = atom_i + (delta / np.linalg.norm(delta)) * hbond_length

    elif n == 2:
        pair_index = np.random.choice(len(pairs))
        atom_i = atoms[pairs[0][pair_index]]
        atom_j = atoms[pairs[1][pair_index]]
        ij = atom_j["coords"] - atom_i["coords"]
        center = (atom_i["coords"] + atom_j["coords"]) / 2
        radius = np.sqrt(hbond_length**2 - (np.linalg.norm(ij) / 2) ** 2)
        normal = ij / np.linalg.norm(ij)

        # Get a basis for the placement circle
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(normal, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        theta = 2 * np.pi * np.random.rand()
        imputed_coords = center + (radius * (np.cos(theta) * u + np.sin(theta) * v))

    elif n == 3:
        pair_index = np.random.choice(len(pairs))
        atom_i = atoms[pairs[0][pair_index]]
        atom_j = atoms[pairs[1][pair_index]]

        # Select third atom
        close_to_i = pairs[1][pairs[0] == pairs[0][pair_index]]
        close_to_j = pairs[1][pairs[0] == pairs[1][pair_index]]
        shared_neighbors = np.setdiff1d(
            np.intersect1d(close_to_i, close_to_j),
            [pairs[0][pair_index], pairs[1][pair_index]],
        )
        if len(shared_neighbors) == 0:
            return None
        atom_k = atoms[np.random.choice(shared_neighbors)]

        circumcenter, circumradius, normal = get_circumcenter(
            atom_i["coords"],
            atom_j["coords"],
            atom_k["coords"],
        )
        if circumcenter is None or hbond_length < circumradius:
            return None
        height = np.sqrt(hbond_length**2 - circumradius**2)
        imputed_coords = circumcenter + height * normal
    else:
        raise ValueError

    noise = np.random.normal(0, noise_std, 3)
    imputed_coords = imputed_coords + noise

    return imputed_coords


def get_circumcenter(a, b, c, epsilon=1e-4):
    """
    Calculates the circumcenter of three points a, b, and c, and returns
    - the circumcenter,
    - the circumradius,
    - the normal vector of the plane containing the three points

    Returns none if the points are collinear.
    """
    u = b - a
    v = c - a
    uxv = np.cross(u, v)

    # Collinearity check
    if np.linalg.norm(uxv) < epsilon:
        return None, None, None

    numerator = np.cross(np.linalg.norm(u) ** 2 * v - np.linalg.norm(v) ** 2 * u, uxv)
    denominator = 2 * np.linalg.norm(uxv) ** 2
    circumcenter = a + numerator / denominator
    circumradius = np.linalg.norm(numerator / denominator)
    normal = uxv / np.linalg.norm(uxv)
    return circumcenter, circumradius, normal
