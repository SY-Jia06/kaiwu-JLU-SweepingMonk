#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Pathfinding helpers for charger override and NPC safety layer.
清扫大作战回桩接管寻路工具 + NPC 安全层。
"""

from collections import deque


DIRS = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

NPC_SAFE_DIST = 2  # Chebyshev safety radius / 切比雪夫安全半径 (E7.3: 3→2)


def _is_traversable(preprocessor, x, z, allow_unknown=False, extra_unknown_cells=None):
    if x < 0 or x >= preprocessor.GRID_SIZE or z < 0 or z >= preprocessor.GRID_SIZE:
        return False

    cell = int(preprocessor.global_map[z, x])
    if cell == 1:
        return False
    if allow_unknown or cell != 0:
        return True
    if extra_unknown_cells is not None and (x, z) in extra_unknown_cells:
        return True
    return (x, z) in preprocessor.charger_cells


def can_move(preprocessor, x, z, dx, dz, allow_unknown=False, extra_unknown_cells=None):
    """Check whether the move is traversable under pessimistic BFS rules."""
    nx, nz = x + dx, z + dz
    if not _is_traversable(preprocessor, nx, nz, allow_unknown=allow_unknown, extra_unknown_cells=extra_unknown_cells):
        return False

    if dx != 0 and dz != 0:
        if not _is_traversable(
            preprocessor, x + dx, z, allow_unknown=allow_unknown, extra_unknown_cells=extra_unknown_cells
        ):
            return False
        if not _is_traversable(
            preprocessor, x, z + dz, allow_unknown=allow_unknown, extra_unknown_cells=extra_unknown_cells
        ):
            return False

    return True


def _build_npc_danger(npcs, radius):
    npc_danger = set()
    for npc in npcs:
        npc_pos = npc.get("pos", {})
        npc_x = int(npc_pos.get("x", 0))
        npc_z = int(npc_pos.get("z", 0))
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                npc_danger.add((npc_x + dx, npc_z + dz))
    return npc_danger


def _charger_cells_from_organ(charger):
    pos = charger.get("pos", {})
    width = int(charger.get("w", 1))
    height = int(charger.get("h", 1))
    base_x = int(pos.get("x", 0))
    base_z = int(pos.get("z", 0))
    return {
        (base_x + dx, base_z + dz)
        for dx in range(width)
        for dz in range(height)
    }


def _charger_access_zone(preprocessor, charger):
    pos = charger.get("pos", {})
    width = int(charger.get("w", 1))
    height = int(charger.get("h", 1))
    base_x = int(pos.get("x", 0))
    base_z = int(pos.get("z", 0))
    zone = set()

    for x in range(base_x - 1, base_x + width + 1):
        for z in range(base_z - 1, base_z + height + 1):
            if x < 0 or x >= preprocessor.GRID_SIZE or z < 0 or z >= preprocessor.GRID_SIZE:
                continue
            if int(preprocessor.global_map[z, x]) == 1:
                continue
            zone.add((x, z))

    return zone


def _bfs_to_targets(
    preprocessor,
    hero_x,
    hero_z,
    target_cells,
    npcs=None,
    npc_block_radius=NPC_SAFE_DIST,
    allow_unknown=False,
    extra_unknown_cells=None,
):
    if npcs is None:
        npcs = []
    if not target_cells:
        return float("inf"), -1

    npc_danger = _build_npc_danger(npcs, npc_block_radius)
    start = (hero_x, hero_z)
    if start in target_cells:
        return 0, -1

    queue = deque()
    visited = {start: None}

    for action_idx, (dx, dz) in enumerate(DIRS):
        if not can_move(
            preprocessor,
            hero_x,
            hero_z,
            dx,
            dz,
            allow_unknown=allow_unknown,
            extra_unknown_cells=extra_unknown_cells,
        ):
            continue
        npos = (hero_x + dx, hero_z + dz)
        if npos in npc_danger or npos in visited:
            continue
        visited[npos] = (start, action_idx)
        if npos in target_cells:
            return 1, action_idx
        queue.append(npos)

    while queue:
        cx, cz = queue.popleft()
        for action_idx, (dx, dz) in enumerate(DIRS):
            if not can_move(
                preprocessor,
                cx,
                cz,
                dx,
                dz,
                allow_unknown=allow_unknown,
                extra_unknown_cells=extra_unknown_cells,
            ):
                continue
            npos = (cx + dx, cz + dz)
            if npos in npc_danger or npos in visited:
                continue
            visited[npos] = ((cx, cz), action_idx)
            if npos in target_cells:
                return _backtrack_first_action(visited, npos, start)
            queue.append(npos)

    return float("inf"), -1


def reachable_charger_distances(preprocessor, hero_x, hero_z, npcs=None, npc_block_radius=NPC_SAFE_DIST):
    """Return sorted reachable charger distances as (dist, center) tuples."""
    if npcs is None:
        npcs = []

    results = []
    chargers = list(getattr(preprocessor, "_chargers", []) or [])
    for charger in chargers:
        target_cells = _charger_cells_from_organ(charger)
        access_zone = _charger_access_zone(preprocessor, charger)
        bfs_dist, bfs_action = _bfs_to_targets(
            preprocessor,
            hero_x,
            hero_z,
            target_cells,
            npcs=npcs,
            npc_block_radius=npc_block_radius,
            allow_unknown=False,
            extra_unknown_cells=access_zone,
        )
        if bfs_action == -1 and bfs_dist != 0:
            continue
        results.append((bfs_dist, preprocessor._charger_center(charger)))

    results.sort(key=lambda item: item[0])
    return results


def bfs_to_charger(preprocessor, hero_x, hero_z, npcs=None, npc_block_radius=NPC_SAFE_DIST, allow_unknown=False):
    """Return BFS distance and first action toward the best reachable charger."""
    if npcs is None:
        npcs = []

    start = (hero_x, hero_z)
    if start in preprocessor.charger_cells:
        return 0, -1

    best_dist = float("inf")
    best_action = -1

    chargers = list(getattr(preprocessor, "_chargers", []) or [])
    if chargers:
        for charger in chargers:
            target_cells = _charger_cells_from_organ(charger)
            access_zone = _charger_access_zone(preprocessor, charger)
            bfs_dist, bfs_action = _bfs_to_targets(
                preprocessor,
                hero_x,
                hero_z,
                target_cells,
                npcs=npcs,
                npc_block_radius=npc_block_radius,
                allow_unknown=allow_unknown,
                extra_unknown_cells=access_zone,
            )
            if bfs_action == -1 and bfs_dist != 0:
                continue
            if bfs_dist < best_dist:
                best_dist = bfs_dist
                best_action = bfs_action

        if best_action != -1 or best_dist == 0:
            return best_dist, best_action

    return _bfs_to_targets(
        preprocessor,
        hero_x,
        hero_z,
        preprocessor.charger_cells,
        npcs=npcs,
        npc_block_radius=npc_block_radius,
        allow_unknown=allow_unknown,
    )


def _backtrack_first_action(visited, target, start):
    """Backtrack from target to start and return (distance, first_action)."""
    path = []
    pos = target
    while pos != start:
        parent, action = visited[pos]
        path.append(action)
        pos = parent

    path.reverse()
    return len(path), path[0]


def _chebyshev(x1, z1, x2, z2):
    return max(abs(x1 - x2), abs(z1 - z2))


def npc_safe_filter(action, hero_x, hero_z, npcs, legal_action, safe_dist=NPC_SAFE_DIST):
    """Post-filter: override action if it moves too close to any NPC."""
    if not npcs:
        return action

    dx, dz = DIRS[action]
    nx, nz = hero_x + dx, hero_z + dz

    is_safe = all(
        _chebyshev(nx, nz, int(npc.get("pos", {}).get("x", 0)), int(npc.get("pos", {}).get("z", 0))) > safe_dist
        for npc in npcs
    )
    if is_safe:
        return action

    best_action = action
    best_min_dist = -1

    for alt in range(8):
        if legal_action[alt] == 0:
            continue
        adx, adz = DIRS[alt]
        anx, anz = hero_x + adx, hero_z + adz
        min_dist = min(
            _chebyshev(anx, anz, int(npc.get("pos", {}).get("x", 0)), int(npc.get("pos", {}).get("z", 0)))
            for npc in npcs
        )
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_action = alt

    return best_action
