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
                if pos not in obstacles:
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
    if angle == 0:
        return is_within_0_degrees(direction, dx, dy)
    elif angle == 90:
        return is_within_90_degrees(direction, dx, dy)
    elif angle == 180:
        return is_within_180_degrees(direction, dx, dy)

def is_within_0_degrees(direction, dx, dy):
    if dx == 0 and dy == 0:
        return False

    # ベクトルの角度を計算
    angle = math.atan2(dy, dx) * 180 / math.pi

    # 方向に基づいて正確な角度を定義
    direction_angles = {
        "north": 90,
        "northeast": 45,
        "east": 0,
        "southeast": -45,
        "south": -90,
        "southwest": -135,
        "west": 180,
        "northwest": 135
    }

    # 角度が方向に厳密に一致するかチェック
    target_angle = direction_angles[direction]
    # 角度を調整して比較
    angle = (angle + 360) % 360
    target_angle = (target_angle + 360) % 360

    return math.isclose(angle, target_angle, abs_tol=1.0)


def is_within_90_degrees(direction, dx, dy):
    if dx == 0 and dy == 0:
        return False

    # 角度を計算
    angle = math.degrees(math.atan2(dx, dy))
    if angle < 0:
        angle += 360

    # 方向とその角度範囲を定義
    direction_angles = {
        "north": (-45, 45),
        "northeast": (0, 90),
        "east": (45, 135),
        "southeast": (90, 180),
        "south": (135, 225),
        "southwest": (180, 270),
        "west": (225, 315),
        "northwest": (-90, 0)
    }

    # 角度範囲を調整し、360度スケールに合わせる
    range_start, range_end = direction_angles[direction]
    if range_start < 0:
        range_start += 360
        range_end += 360
        if range_end >= 360:
            return (0 <= angle <= range_end % 360) or (range_start <= angle <= 360)
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
        "north": (-90, 90),
        "northeast": (-45, 135),
        "east": (0, 180),
        "southeast": (45, 225),
        "south": (90, 270),
        "southwest": (135, 315),
        "west": (180, 360),
        "northwest": (-135, 45)
    }

    # 角度範囲を調整し、360度スケールに合わせる
    range_start, range_end = direction_angles[direction]
    if range_start < 0:
        range_start += 360
        range_end += 360
        if range_end >= 360:
            return (0 <= angle <= range_end % 360) or (range_start <= angle <= 360)
    return range_start <= angle <= range_end

def is_line_blocked_by_obstacles(start, end, obstacles):
    """Check if a line from start to end is blocked by any obstacle."""
    points = supercover_line(start[0], start[1], end[0], end[1])
    for point in points:
        if (point[0], point[1]) in obstacles:
            return True
    return False


def supercover_line(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    num_steps = max(abs(dx), abs(dy))
    dx = dx / num_steps
    dy = dy / num_steps

    x = x0
    y = y0
    for _ in range(num_steps + 1):
        points.append((int(round(x)), int(round(y))))
        x += dx
        y += dy

    return set(points)  # 重複を避けるためにセットに変換




def visualize_camera_coverage_by_position(grid_size, x, coverage, obstacles):
    fig, ax = plt.subplots(figsize=(2*grid_size[0], 2*grid_size[1]))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
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
            ax.text(pos[0] + 0.5, grid_size[1] - pos[1] - 0.5, f'{cam_type}\n{direction}', color='black', ha='center', va='center')

    plt.gca().invert_yaxis()  # y軸を反転
    plt.legend()
    plt.savefig("camera_placement_by_position.png")


# この関数を呼び出す際には、実際の grid_size, x (解の変数), および coverage データを提供する必要があります。

def calculate(positions, directions, coverage, camera_types, grid_size, budget, obstacles):
    # PuLP問題の設定
    prob = pulp.LpProblem("CameraPlacement", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ((pos, cam_type, dir) for pos in positions for cam_type in camera_types for dir in directions[cam_type]), cat="Binary")
    z = pulp.LpVariable.dicts("z", ((i, j) for i in range(grid_size[0]) for j in range(grid_size[1])), cat="Binary")

    # 目的関数
    prob += pulp.lpSum([z[(i, j)] for i in range(grid_size[0]) for j in range(grid_size[1])])

    # 予算制約
    prob += pulp.lpSum(camera_types[cam_type] * x[(pos, cam_type, dir)] for pos in positions for cam_type in camera_types for dir in directions[cam_type]) <= budget

    # 配置制約
    for pos in positions:
        prob += pulp.lpSum([x[(pos, cam_type, dir)] for cam_type in camera_types for dir in directions[cam_type]]) <= 1

    # カバレッジ制約
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 各グリッドセル (i, j) がカバーされていることを確認
            prob += z[(i, j)] <= pulp.lpSum(x[pos, cam_type, dir] for pos in positions for cam_type in camera_types for dir in directions[cam_type] if (i, j) in coverage[pos][cam_type][dir])

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



def main():
    # Grid size and camera setup
    grid_size = (10, 8)  # Example grid size
    positions = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
    directions = {
        "A": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "B": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "C": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
    }
    budget = 1200
    camera_angle = {"A": 0, "B": 90, "C": 180}  # Camera angles in degrees
    camera_distance = {"A": 4, "B": 3, "C": 2}  # Maximum observation distance
    obstacles = {
        (1, 1),(2, 1), (3, 1), (4, 1), (5, 1), 
        (2, 4), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
        (6, 6), (7, 6), (8, 6), (9, 6), (6, 8),
        }
    coverage = compute_coverage(positions, directions, grid_size, camera_distance, obstacles, camera_angle)
    calculate(positions, directions, coverage, camera_types={"A": 100, "B": 200, "C": 500}, grid_size=grid_size, budget=budget, obstacles=obstacles)

if __name__ == "__main__":
    main()