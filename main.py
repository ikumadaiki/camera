import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pulp


def compute_coverage(
    positions, directions, grid_size, camera_distance, obstacles, camera_angle
):
    coverage = {}
    for pos in positions:
        coverage[pos] = {}
        for cam_type, _ in camera_angle.items():
            coverage[pos][cam_type] = {}
            for dir in directions:
                coverage[pos][cam_type][dir] = []
                if pos not in obstacles:
                    coverage[pos][cam_type][dir].append(pos)
                for dx in range(
                    -camera_distance[cam_type], camera_distance[cam_type] + 1
                ):
                    for dy in range(
                        -camera_distance[cam_type], camera_distance[cam_type] + 1
                    ):
                        if dx == 0 and dy == 0:
                            continue
                        new_col = pos[0] + dx
                        new_row = pos[1] + dy
                        if 0 <= new_col < grid_size[0] and 0 <= new_row < grid_size[1]:
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= camera_distance[cam_type]:
                                if is_within_degrees(
                                    dir, dx, dy, camera_angle[cam_type]
                                ):
                                    if not is_line_blocked_by_obstacles(
                                        pos, (new_col, new_row), obstacles
                                    ):
                                        coverage[pos][cam_type][dir].append(
                                            (new_col, new_row)
                                        )
    import pdb

    pdb.set_trace()
    return coverage


def is_within_degrees(direction, dx, dy, angle):
    if angle == 90:
        return is_within_90_degrees(direction, dx, dy)
    elif angle == 180:
        return is_within_180_degrees(direction, dx, dy)
    elif angle == 360:
        return is_within_360_degrees(direction, dx, dy)


def is_within_90_degrees(direction, dx, dy):
    if dx == 0 and dy == 0:
        return False

    # 角度を計算
    angle = math.degrees(math.atan2(dx, dy))
    angle = (angle + 360) % 360  # 角度を正の範囲で正規化

    # 方向とその角度範囲を定義（より直感的な定義に）
    direction_angles = {
        "north": (315, 45),
        "northeast": (0, 90),
        "east": (45, 135),
        "southeast": (90, 180),
        "south": (135, 225),
        "southwest": (180, 270),
        "west": (225, 315),
        "northwest": (270, 360),
    }

    range_start, range_end = direction_angles[direction]
    if range_end < range_start:
        return (0 <= angle <= range_end) or (range_start <= angle <= 360)
    return range_start <= angle <= range_end


def is_within_180_degrees(direction, dx, dy):
    if dx == 0 and dy == 0:
        return False

    # 角度を計算
    angle = math.degrees(math.atan2(dx, dy))
    if angle < 0:
        angle += 360

    # 方向とその角度範囲を定義
    direction_angles = {
        "north": (270, 90),
        "northeast": (315, 135),
        "east": (0, 180),
        "southeast": (45, 225),
        "south": (90, 270),
        "southwest": (135, 315),
        "west": (180, 360),
        "northwest": (225, 45),
    }

    # 角度範囲を調整し、360度スケールに合わせる
    range_start, range_end = direction_angles[direction]
    if range_end < range_start:
        return (0 <= angle <= range_end) or (range_start <= angle <= 360)
    return range_start <= angle <= range_end


def is_within_360_degrees(direction, dx, dy):
    if dx == 0 and dy == 0:
        return False

    # 角度を計算
    angle = math.degrees(math.atan2(dx, dy))
    if angle < 0:
        angle += 360

    # 方向とその角度範囲を定義
    direction_angles = {
        "north": (0, 360),
        "northeast": (0, 360),
        "east": (0, 360),
        "southeast": (0, 360),
        "south": (0, 360),
        "southwest": (0, 360),
        "west": (0, 360),
        "northwest": (0, 360),
    }

    # 角度範囲を調整し、360度スケールに合わせる
    range_start, range_end = direction_angles[direction]
    if range_end < range_start:
        return (0 <= angle <= range_end) or (range_start <= angle <= 360)
    return range_start <= angle <= range_end


def is_line_blocked_by_obstacles(start, end, obstacles):
    """Check if a line from start to end is blocked by any obstacle."""
    points = supercover_line_(start[0], start[1], end[0], end[1])
    for point in points:
        if (point[0], point[1]) in obstacles:
            return True
    return False


def supercover_line_(x0, y0, x1, y1):
    points = set()
    eps = 1e-6
    dx = x1 - x0
    dy = y1 - y0
    num_steps = max(abs(dx), abs(dy))
    x = x0
    y = y0
    dx = dx / num_steps
    dy = dy / num_steps
    for i in range(2 * num_steps + 1):
        if i == 0:
            points.add((x, y))
            x += 0.5 * dx
            y += 0.5 * dy
        if i > 0:
            if x - int(x) == 0 and y - int(y) == 0:  # パターン1
                points.add((x, y))
            elif x - int(x) == 0.5 and y - int(y) == 0.5:  # パターン2
                pass
            elif x - int(x) != 0.5 or y - int(y) != 0.5:
                if abs(dx) >= abs(dy):
                    if (
                        abs(x - int(x) - 0.5) < eps and abs(y - int(y) - 0.5) < eps
                    ):  # パターン2
                        pass
                    elif abs(y - int(y) - 0.5) < eps:  # パターン3
                        points.add((round(x), round(y + eps)))
                        points.add((round(x), round(y - eps)))
                    else:  # パターン4
                        points.add((math.floor(x), round(y)))
                        points.add((math.ceil(x), round(y)))
                elif abs(dx) < abs(dy):
                    if (
                        abs(x - int(x) - 0.5) < eps and abs(y - int(y) - 0.5) < eps
                    ):  # パターン2
                        pass
                    elif abs(x - int(x) - 0.5) < eps:  # パターン3
                        points.add((round(x + eps), round(y)))
                        points.add((round(x - eps), round(y)))
                    else:  # パターン4
                        points.add((round(x), math.floor(y)))
                        points.add((round(x), math.ceil(y)))
            x += 0.5 * dx
            y += 0.5 * dy

    return set(points)


def visualize_camera_coverage_by_position(
    grid_size, x, coverage, obstacles, camera_angles
):
    fig, ax = plt.subplots(figsize=(2 * grid_size[0], 2 * grid_size[1]))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks([i + 0.5 for i in range(grid_size[0])])  # Offset to center labels
    ax.set_yticks(
        [i - 0.5 for i in range(grid_size[1], 0, -1)]
    )  # Offset to center labels
    ax.set_xticklabels(
        [str(num) for num in range(grid_size[0])],
        fontsize=30,
        verticalalignment="center",
    )
    ax.set_yticklabels(
        [str(num) for num in range(grid_size[1])],
        fontsize=30,
        horizontalalignment="right",
    )  # ax.grid(True)

    # グリッドの描画
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle(
                (j, grid_size[1] - i - 1),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    # 障害物の描画
    for obstacle in obstacles:
        rect = patches.Rectangle(
            (obstacle[0], grid_size[1] - obstacle[1] - 1),
            1,
            1,
            linewidth=1,
            edgecolor="r",
            facecolor="black",
        )
        ax.add_patch(rect)

    # 色のマッピングを生成
    color_map = {"A": "red", "B": "blue", "C": "green", "D": "purple"}

    # カメラの配置とカバレッジ領域の表示
    for key, var in x.items():
        if var.varValue > 0.5:  # このカメラが配置されている
            pos, cam_type, direction = key
            if camera_angles[cam_type] == 360:
                direction = ""
            camera_color = color_map[cam_type]
            # カバレッジ領域の表示
            for i, j in coverage[key[0]][key[1]][key[2]]:
                rect = patches.Rectangle(
                    (i, grid_size[1] - j - 1),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="none",
                    facecolor=camera_color,
                    alpha=0.5,
                )
                ax.add_patch(rect)
            ax.text(
                pos[0] + 0.5,
                grid_size[1] - pos[1] - 0.5,
                f"{cam_type}\n{direction}",
                color="black",
                ha="center",
                va="center",
                fontsize=20,
            )

    plt.gca().invert_yaxis()  # y軸を反転
    plt.legend()
    plt.savefig("camera_placement_by_position.png")


# この関数を呼び出す際には、実際の grid_size, x (解の変数), および coverage データを提供する必要があります。


def calculate(
    positions,
    directions,
    coverage,
    camera_costs,
    grid_size,
    budget,
    obstacles,
    camera_angles,
    necessary_area,
):
    # PuLP問題の設定
    prob = pulp.LpProblem("CameraPlacement", pulp.LpMaximize)
    x = pulp.LpVariable.dicts(
        "x",
        (
            (pos, cam_type, dir)
            for pos in positions
            for cam_type in camera_costs
            for dir in directions
        ),
        cat="Binary",
    )
    z = pulp.LpVariable.dicts(
        "z",
        ((i, j) for i in range(grid_size[0]) for j in range(grid_size[1])),
        cat="Binary",
    )
    # 目的関数
    prob += pulp.lpSum(
        [z[(i, j)] for i in range(grid_size[0]) for j in range(grid_size[1])]
    )
    # 予算制約
    prob += (
        pulp.lpSum(
            camera_costs[cam_type] * x[(pos, cam_type, dir)]
            for pos in positions
            for cam_type in camera_costs
            for dir in directions
        )
        <= budget
    )
    # 配置制約
    for pos in positions:
        prob += (
            pulp.lpSum(
                [
                    x[(pos, cam_type, dir)]
                    for cam_type in camera_costs
                    for dir in directions
                ]
            )
            <= 1
        )
    # カバレッジ制約
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 各グリッドセル (i, j) がカバーされていることを確認
            prob += z[(i, j)] <= pulp.lpSum(
                x[pos, cam_type, dir]
                for pos in positions
                for cam_type in camera_costs
                for dir in directions
                if (i, j) in coverage[pos][cam_type][dir]
            )
    for pos in necessary_area:
        prob += (
            pulp.lpSum(
                [
                    z[(i, j)]
                    for i in range(pos[0], pos[0] + 1)
                    for j in range(pos[1], pos[1] + 1)
                ]
            )
            == 1
        )
    # 問題を解く
    prob.solve()

    if pulp.LpStatus[prob.status] == "Optimal":
        print("========================================")
        for v in prob.variables():
            if v.varValue > 0:
                print(v.name, "=", v.varValue)
        print("Optimal value:", pulp.value(prob.objective))

        # カメラの配置とカバレッジ領域の可視化
        visualize_camera_coverage_by_position(
            grid_size, x, coverage, obstacles, camera_angles
        )


# 指定したxで可視化
def visualize_camera_coverage_by_x(
    grid_size, coverage, pos, cam_type, direction, obstacles
):
    plt.clf()
    fig, ax = plt.subplots(figsize=(2 * grid_size[0], 2 * grid_size[1]))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.set_xticklabels([str(num) for num in range(grid_size[0] + 1)], fontsize=50)
    ax.set_yticklabels(
        [str(num) for num in reversed(range(grid_size[1] + 1))], fontsize=50
    )
    ax.grid(True)

    # グリッドの描画
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle(
                (j, grid_size[1] - i - 1),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    # 障害物の描画
    for obstacle in obstacles:
        rect = patches.Rectangle(
            (obstacle[0], grid_size[1] - obstacle[1] - 1),
            1,
            1,
            linewidth=1,
            edgecolor="r",
            facecolor="black",
        )
        ax.add_patch(rect)

    # 色のマッピングを生成
    # unique_positions = set(key[0] for key in x.keys() if x[key].varValue > 0.5)
    # color_map = {pos: plt.cm.get_cmap('viridis')(i / len(unique_positions)) for i, pos in enumerate(unique_positions)}

    # カメラの配置とカバレッジ領域の表示
    camera_color = "red"
    # カバレッジ領域の表示
    for i, j in coverage[pos][cam_type][direction]:
        rect = patches.Rectangle(
            (i, grid_size[1] - j - 1),
            1,
            1,
            linewidth=1,
            edgecolor="none",
            facecolor=camera_color,
            alpha=0.5,
        )
        ax.add_patch(rect)
    ax.text(
        pos[0] + 0.5,
        grid_size[1] - pos[1] - 0.5,
        f"{cam_type}\n{direction}",
        color="black",
        ha="center",
        va="center",
    )

    plt.gca().invert_yaxis()  # y軸を反転
    plt.legend()
    plt.savefig("camera_placement_by_position_x.png")


def cam_possible(pos, obstacles):
    i, j = pos
    # チェックするべき隣接位置をリスト化
    neighbors = [
        (0, 0),  # 現在の位置
        (1, 0),  # 右
        (0, 1),  # 下
        (1, 1),  # 右下
        (-1, 0),  # 左
        (0, -1),  # 上
        (-1, -1),  # 左上
        (1, -1),  # 右上
        (-1, 1),  # 左下
    ]

    # どの隣接位置も障害物を含まないかチェック
    for di, dj in neighbors:
        if (i + di, j + dj) in obstacles:
            return False
    return True


def main():
    grid_size = (10, 10)  # エリアのサイズ
    directions = [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]
    camera_distance = {"A": 4, "B": 1, "C": 2, "D": 1}  # Camera angles in degrees
    camera_angle = {
        "A": 90,
        "B": 180,
        "C": 360,
        "D": 360,
    }  # Maximum observation distance
    camera_costs = {"A": 33, "B": 12, "C": 30, "D": 21}
    budget = 100
    obstacles = {
        (1, 3),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 1),
        (2, 8),
        (2, 9),
        (3, 1),
        (3, 3),
        (3, 6),
        (3, 8),
        (3, 9),
        (4, 1),
        (4, 3),
        (4, 5),
        (4, 6),
        (4, 8),
        (4, 9),
        (5, 1),
        (5, 5),
        (5, 6),
        (5, 8),
        (5, 9),
        (6, 1),
        (6, 8),
        (6, 9),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 5),
        (7, 6),
        (7, 8),
        (7, 9),
        (8, 1),
        (8, 2),
        (8, 3),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 8),
        (9, 9),
    }
    necessary_area = {(0, 3)}
    positions = [
        (i, j)
        for i in range(grid_size[0])
        for j in range(grid_size[1])
        if not cam_possible((i, j), obstacles)
    ]
    coverage = compute_coverage(
        positions, directions, grid_size, camera_distance, obstacles, camera_angle
    )
    calculate(
        positions,
        directions,
        coverage,
        camera_costs=camera_costs,
        grid_size=grid_size,
        budget=budget,
        obstacles=obstacles,
        camera_angles=camera_angle,
        necessary_area=necessary_area,
    )


if __name__ == "__main__":
    main()
