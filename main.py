import matplotlib.pyplot as plt
import numpy as np
import pulp
import matplotlib.patches as patches
import math

def compute_coverage(positions, directions, grid_size, camera_distance, obstacles, camera_angle):
    coverage = {}
    for pos in positions:
        coverage[pos] = {}
        for cam_type, cam_dirs in directions.items():
            coverage[pos][cam_type] = {}
            for dir in cam_dirs:
                coverage[pos][cam_type][dir] = []
                if pos not in obstacles and camera_angle[cam_type]==360:
                    coverage[pos][cam_type][dir].append(pos)
                for dx in range(-camera_distance[cam_type], camera_distance[cam_type] + 1):
                    for dy in range(-camera_distance[cam_type], camera_distance[cam_type] + 1):
                        if dx == 0 and dy == 0:
                            continue 
                        new_col = pos[0] + dx
                        new_row = pos[1] + dy
                        if 0 <= new_col < grid_size[0] and 0 <= new_row < grid_size[1]:
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= camera_distance[cam_type]:
                                if is_within_degrees(dir, dx, dy, camera_angle[cam_type]):
                                    if not is_line_blocked_by_obstacles(pos, (new_col, new_row), obstacles):
                                        coverage[pos][cam_type][dir].append((new_col, new_row))
    import pdb; pdb.set_trace()
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
        "northwest": (270, 360)
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
        "northwest": (225, 45)
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
        "northwest": (0, 360)
    }

    # 角度範囲を調整し、360度スケールに合わせる
    range_start, range_end = direction_angles[direction]
    if range_end < range_start:
        return (0 <= angle <= range_end) or (range_start <= angle <= 360)
    return range_start <= angle <= range_end


def is_line_blocked_by_obstacles(start, end, obstacles):
    """Check if a line from start to end is blocked by any obstacle."""
    points = supercover_line(start[0], start[1], end[0], end[1])
    if points != supercover_line_(start[0], start[1], end[0], end[1]):
        import pdb; pdb.set_trace()
        supercover_line_(start[0], start[1], end[0], end[1])
    for point in points:
        if (point[0], point[1]) in obstacles:
            return True
    return False


def supercover_line(x0, y0, x1, y1):
    points = []
    eps = 1e-6
    dx = x1 - x0
    dy = y1 - y0
    num_steps_x = abs(dx)
    num_steps_y = abs(dy)
    x = x0
    y = y0
    dx = dx / num_steps_x if num_steps_x != 0 else 0
    dy = dy / num_steps_x if num_steps_x != 0 else num_steps_y
    # if (x0, y0, x1, y1) == (0, 1, 3, 0):
    #     import pdb; pdb.set_trace()
    for i in range(2 * num_steps_x + 1):
        if i == 0:
            points.append((x, y))
            points.append((x1, y1))
            x += 0.5 * dx
            y += 0.5 * dy       
        if i > 0:
            if abs(y - int(y) - 0.5) < eps:
                pass
            elif y - int(y) != 0.5:
                if dx < 0:
                    points.append((math.floor(x), round(y)))
                else:
                    points.append((math.ceil(x), round(y)))
            elif x - int(x) == 0 and y - int(y) == 0:
                points.append((int(round(x)), int(round(y))))
            x += 0.5 * dx
            y += 0.5 * dy
    dx = x1 - x0
    dy = y1 - y0
    x = x0
    y = y0
    dx = dx / num_steps_y if num_steps_y != 0 else num_steps_x
    dy = dy / num_steps_y if num_steps_y != 0 else 0
    for i in range(2 * num_steps_y + 1):
        if i == 0:
            x += 0.5 * dx
            y += 0.5 * dy
        if i > 0:
            if abs(x - int(x) - 0.5) < eps:
                pass
            elif x - int(x) != 0.5:
                if dy < 0:
                    points.append((round(x), math.floor(y)))
                else:
                    points.append((round(x), math.ceil(y)))
            elif x - int(x) == 0 and y - int(y) == 0:
                points.append((int(round(x)), int(round(y))))
            x += 0.5 * dx
            y += 0.5 * dy

    return set(points)

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
            if x - int(x) == 0 and y - int(y) == 0: # パターン1
                points.add((x, y))
            elif x - int(x) == 0.5 and y - int(y) == 0.5: # パターン2
                pass
            elif x - int(x) != 0.5 or y - int(y) != 0.5:
                if abs(dx) >= abs(dy):
                    if abs(x - int(x) - 0.5) < eps and abs(y - int(y) - 0.5) < eps: # パターン2
                        pass
                    elif abs(y - int(y) - 0.5) < eps: # パターン3
                        points.add((round(x), round(y+eps)))
                        points.add((round(x), round(y-eps)))
                    else: # パターン4
                        points.add((math.floor(x), round(y)))
                        points.add((math.ceil(x), round(y)))
                elif abs(dx) < abs(dy):
                    if abs(x - int(x) - 0.5) < eps and abs(y - int(y) - 0.5) < eps: # パターン2
                        pass
                    elif abs(x - int(x) - 0.5) < eps: # パターン3
                        points.add((round(x+eps), round(y)))
                        points.add((round(x-eps), round(y)))
                    else: # パターン4
                        points.add((round(x), math.floor(y)))
                        points.add((round(x), math.ceil(y)))
            x += 0.5 * dx
            y += 0.5 * dy

    return set(points)




def visualize_camera_coverage_by_position(grid_size, x, coverage, obstacles):
    fig, ax = plt.subplots(figsize=(2*grid_size[0], 2*grid_size[1]))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks([i + 0.5 for i in range(grid_size[0])])  # Offset to center labels
    ax.set_yticks([i - 0.5 for i in range(grid_size[1], 0, -1)])  # Offset to center labels
    ax.set_xticklabels([str(num) for num in range(grid_size[0])], fontsize=30, verticalalignment='center')
    ax.set_yticklabels([str(num) for num in range(grid_size[1])], fontsize=30, horizontalalignment='right')    # ax.grid(True)

    # グリッドの描画
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle((j, grid_size[1] - i - 1), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # 障害物の描画
    for obstacle in obstacles:
        rect = patches.Rectangle((obstacle[0], grid_size[1] - obstacle[1] - 1), 1, 1, linewidth=1, edgecolor='r', facecolor='black')
        ax.add_patch(rect)
    
    # 色のマッピングを生成
    unique_positions = set(key[0] for key in x.keys() if x[key].varValue > 0.5)
    color_map = {pos: plt.cm.get_cmap('viridis')(i / len(unique_positions)) for i, pos in enumerate(unique_positions)}

    # カメラの配置とカバレッジ領域の表示
    for key, var in x.items():
        if var.varValue > 0.5:  # このカメラが配置されている
            pos, cam_type, direction = key
            camera_color = color_map[pos]
            # カバレッジ領域の表示
            for (i, j) in coverage[key[0]][key[1]][key[2]]:
                rect = patches.Rectangle((i, grid_size[1] - j - 1), 1, 1, linewidth=1, edgecolor='none', facecolor=camera_color, alpha=0.5)
                ax.add_patch(rect)
            ax.text(pos[0] + 0.5, grid_size[1] - pos[1] - 0.5, f'{cam_type}\n{direction}', color='black', ha='center', va='center', fontsize=20)

    plt.gca().invert_yaxis()  # y軸を反転
    plt.legend()
    plt.savefig("camera_placement_by_position.png")


# この関数を呼び出す際には、実際の grid_size, x (解の変数), および coverage データを提供する必要があります。

def calculate(positions, directions, coverage, camera_types, grid_size, budget, obstacles):
    # PuLP問題の設定
    # prob = pulp.LpProblem("CameraPlacement", pulp.LpMaximize)
    prob = pulp.LpProblem("CameraPlacement", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((pos, cam_type, dir) for pos in positions for cam_type in camera_types for dir in directions[cam_type]), cat="Binary")
    # z = pulp.LpVariable.dicts("z", ((i, j) for i in range(grid_size[0]) for j in range(grid_size[1])), cat="Binary")

    for pos in positions:
        for cam_type in camera_types:
            for dir in directions[cam_type]:
                x[pos, cam_type, dir] = pulp.LpVariable(f"x_{pos}_{cam_type}_{dir}", cat="Binary")

    # 目的関数
    # prob += pulp.lpSum([z[(i, j)] for i in range(grid_size[0]) for j in range(grid_size[1])])
    prob += pulp.lpSum(camera_types[cam_type] * x[(pos, cam_type, dir)] for pos in positions for cam_type in camera_types for dir in directions[cam_type])

    # 予算制約
    # prob += pulp.lpSum(camera_types[cam_type] * x[(pos, cam_type, dir)] for pos in positions for cam_type in camera_types for dir in directions[cam_type]) <= budget

    # 配置制約
    for pos in positions:
        prob += pulp.lpSum([x[(pos, cam_type, dir)] for cam_type in camera_types for dir in directions[cam_type]]) <= 1

    # カバレッジ制約
    for area in positions:
        if area in obstacles:
            continue
            # 各グリッドセル (i, j) がカバーされていることを確認
            # prob += z[(i, j)] <= pulp.lpSum(x[pos, cam_type, dir] for pos in positions for cam_type in camera_types for dir in directions[cam_type] if (i, j) in coverage[pos][cam_type][dir])
        prob += 1 <= pulp.lpSum(x[pos, cam_type, dir] for pos in positions for cam_type in camera_types for dir in directions[cam_type] if area in coverage[pos][cam_type][dir])

    # 問題を解く
    prob.solve()

    if pulp.LpStatus[prob.status] == "Optimal":
        print("========================================")
        for v in prob.variables():
            if v.varValue > 0:
                print(v.name, "=", v.varValue)
        print("Optimal value:", pulp.value(prob.objective))

        # カメラの配置とカバレッジ領域の可視化
        visualize_camera_coverage_by_position(grid_size, x, coverage, obstacles)

# 指定したxで可視化
def visualize_camera_coverage_by_x(grid_size, coverage, pos, cam_type, direction, obstacles):
    plt.clf()
    fig, ax = plt.subplots(figsize=(2*grid_size[0], 2*grid_size[1]))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.set_xticklabels([str(num) for num in range(grid_size[0] + 1)], fontsize=50)
    ax.set_yticklabels([str(num) for num in reversed(range(grid_size[1] + 1))], fontsize=50)
    ax.grid(True)

    # グリッドの描画
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            rect = patches.Rectangle((j, grid_size[1] - i - 1), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # 障害物の描画
    for obstacle in obstacles:
        rect = patches.Rectangle((obstacle[0], grid_size[1] - obstacle[1] - 1), 1, 1, linewidth=1, edgecolor='r', facecolor='black')
        ax.add_patch(rect)
    
    # 色のマッピングを生成
    # unique_positions = set(key[0] for key in x.keys() if x[key].varValue > 0.5)
    # color_map = {pos: plt.cm.get_cmap('viridis')(i / len(unique_positions)) for i, pos in enumerate(unique_positions)}

    # カメラの配置とカバレッジ領域の表示
    camera_color = "red"
    # カバレッジ領域の表示
    for (i, j) in coverage[pos][cam_type][direction]:
        rect = patches.Rectangle((i, grid_size[1] - j - 1), 1, 1, linewidth=1, edgecolor='none', facecolor=camera_color, alpha=0.5)
        ax.add_patch(rect)
    ax.text(pos[0] + 0.5, grid_size[1] - pos[1] - 0.5, f'{cam_type}\n{direction}', color='black', ha='center', va='center')

    plt.gca().invert_yaxis()  # y軸を反転
    plt.legend()
    plt.savefig("camera_placement_by_position_x.png")




def main():
    grid_size = (10, 10)  # エリアのサイズ
    positions = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
    directions = {
        "A": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "B": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "C": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "D": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        # "E": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
    }
    camera_distance = {"A": 4, "B": 1, "C": 2, "D": 1}  # Camera angles in degrees
    camera_angle = {"A": 90, "B": 180, "C": 360, "D": 360}  # Maximum observation distance
    camera_types={"A": 33,"B": 12, "C": 30, "D": 21}
    budget = 150
    obstacles = {
        (1, 1),(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1),
        (2, 4), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28),
        (6, 6), (7, 6), (8, 6), (9, 6), (6, 8), (7, 8), (8, 8), (9, 8),
        (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18),
        (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18),
        (11, 11), (12, 11), (13, 11), (14, 11), (15, 11), (16, 11), (17, 11), (18, 11),
        (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18),
        (16, 6), (16, 7), (16, 8), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (16, 16), (16, 17), (16, 18),
        (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (19, 11), (19, 12), (19, 13), (19, 14), (19, 15), (19, 16), (19, 17), (19, 18),
        (26, 8), (26, 9), (26, 10), (26, 11), (26, 12), (26, 13), (26, 14), (26, 15), (26, 16), (26, 17), (26, 18),
        (21, 21), (22, 21), (23, 21), (24, 21), (25, 21), (26, 21), (27, 21), (28, 21), (29, 21),
        (21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (21, 28), (21, 29),
        (26, 26), (26, 27), (26, 28), (26, 29),
        (28, 6), (28, 7), (28, 8), (28, 9), (28, 10), (28, 11), (28, 12), (28, 13), (28, 14), (28, 15), (28, 16), (28, 17), (28, 18),
        (29, 2), (29, 3), (29, 4), (29, 5), (29, 6), (29, 7), (29, 8), (29, 9), (29, 10), (29, 11), (29, 12), (29, 13), (29, 14), (29, 15), (29, 16), (29, 17), (29, 18),
        (16, 21), (17, 21), (18, 21), (19, 21), (20, 21), (21, 21), (22, 21), (23, 21), (24, 21), (25, 21),
        (16, 22), (16, 23), (16, 24), (16, 25), (16, 26), (16, 27), (16, 28), (16, 29),
        (21, 26), (21, 27), (21, 28),
        (24, 6), (24, 7), (24, 8), (24, 9), (24, 10), (24, 11), (24, 12), (24, 13), (24, 14), (24, 15), (24, 16), (24, 17), (24, 18),
        (26, 16), (26, 17), (26, 18),
        (28, 21), (29, 21),
        (28, 22), (28, 23), (28, 24), (28, 25), (28, 26), (28, 27), (28, 28), (28, 29),
        (29, 26), (29, 27), (29, 28), (29, 29),
        }
    obstacles = {(1, 3), (1, 7), (1, 8), (1, 9), (2, 1), (2, 8), (2, 9), (3, 1), (3, 3), (3, 6), (3, 8), (3, 9), (4, 1), (4, 3), (4, 5), (4, 6), (4, 8), (4, 9), (5, 1), (5, 5), (5, 6), (5, 8), (5, 9), (6, 1), (6, 8), (6, 9), (7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 8), (7, 9), (8, 1), (8, 2), (8, 3), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 8), (9, 9)}
    coverage = compute_coverage(positions, directions, grid_size, camera_distance, obstacles, camera_angle)
    calculate(positions, directions, coverage, camera_types, grid_size=grid_size, budget=budget, obstacles=obstacles)
    pos, cam_type, direction = (3, 7), "A", "southwest"
    import pdb; pdb.set_trace()
    visualize_camera_coverage_by_x(grid_size, coverage, pos, cam_type, direction, obstacles)


if __name__ == "__main__":
    main()