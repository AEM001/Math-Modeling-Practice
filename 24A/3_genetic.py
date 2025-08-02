## 问题三代码 - 遗传算法优化版
"""
舞龙队伍调头空间螺距优化分析 - 采用遗传算法优化搜索
目标：找到在调头空间外不会发生碰撞的最小螺距
"""

import numpy as np
import random
from typing import List, Tuple

def binary_search_zero(func, left_bound, right_bound, tolerance, *args):
    """二分法求零点 - 高效数值求解"""
    while abs(right_bound - left_bound) > tolerance:
        mid_point = (left_bound + right_bound) / 2
        if func(mid_point, *args) == 0:
            return mid_point
        elif func(left_bound, *args) * func(mid_point, *args) < 0:
            right_bound = mid_point
        else:
            left_bound = mid_point
    return (left_bound + right_bound) / 2

def calculate_dragon_positions(start_theta, spiral_pitch, max_segments=223):

    # 物理参数
    DRAGON_RADIUS = 0.275
    FIRST_SEGMENT_DISTANCE = 3.41 - DRAGON_RADIUS * 2
    REGULAR_SEGMENT_DISTANCE = 2.2 - DRAGON_RADIUS * 2
    
    theta_positions = [start_theta]
    
    for i in range(max_segments):
        segment_distance = FIRST_SEGMENT_DISTANCE if i == 0 else REGULAR_SEGMENT_DISTANCE
        previous_theta = theta_positions[-1]
        
        # 几何约束方程：相邻节点间的距离约束
        def distance_constraint(theta):
            return (theta**2 + previous_theta**2 - 
                   2*theta*previous_theta*np.cos(theta - previous_theta) - 
                   4*np.pi**2*segment_distance**2/spiral_pitch**2)
        
        # 求解下一个节点位置
        next_theta = binary_search_zero(
            distance_constraint, 
            previous_theta, 
            previous_theta + np.pi/2, 
            1e-8
        )
        theta_positions.append(next_theta)
        
        # 三圈终止条件
        if next_theta - start_theta >= 3*np.pi:
            break
    
    return theta_positions

def convert_to_cartesian(theta_positions, spiral_pitch):
    """将极坐标转换为笛卡尔坐标"""
    coordinates = []
    for theta in theta_positions:
        radius = theta * spiral_pitch / (2 * np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        coordinates.append((x, y))
    return coordinates

def calculate_path_slopes(coordinates):
    """计算路径各段的斜率"""
    slopes = []
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        slope = (y1 - y2) / (x1 - x2)
        slopes.append(slope)
    return slopes

def check_collision_with_benches(start_theta, coordinates, slopes, theta_positions):
    """
    检测与板凳的碰撞 - 使用几何光学反射模型
    检查四个关键反射路径
    """
    DRAGON_RADIUS = 0.275
    BENCH_RADIUS = 0.15
    
    def calculate_reflection_path(base_slope, base_x, base_y, reflection_type, target_x, target_y):
        """计算反射路径"""
        # 反射系数计算
        if reflection_type == "positive":
            reflected_slope = (BENCH_RADIUS/DRAGON_RADIUS + base_slope) / (1 - BENCH_RADIUS*base_slope/DRAGON_RADIUS)
        else:
            reflected_slope = (base_slope - BENCH_RADIUS/DRAGON_RADIUS) / (1 + BENCH_RADIUS*base_slope/DRAGON_RADIUS)
        
        # 计算截距
        intercept = BENCH_RADIUS*np.sqrt(base_slope**2 + 1) + base_y - base_slope*base_x
        if np.abs(intercept) <= np.abs(base_y - base_slope*base_x):
            intercept = -BENCH_RADIUS*np.sqrt(base_slope**2 + 1) + base_y - base_slope*base_x
        
        # 计算交点
        if reflection_type == "positive":
            intersection_x = (base_y - reflected_slope*base_x - intercept) / (base_slope - reflected_slope)
            intersection_y = (base_slope*base_y - base_slope*reflected_slope*base_x - reflected_slope*intercept) / (base_slope - reflected_slope)
        else:
            intersection_x = (target_y - reflected_slope*target_x - intercept) / (base_slope - reflected_slope)
            intersection_y = (base_slope*target_y - base_slope*reflected_slope*target_x - intercept*reflected_slope) / (base_slope - reflected_slope)
        
        return intersection_x, intersection_y
    
    # 检查四个
    collision_cases = [
        (0, 0, "positive", 1), 
        (0, 0, "negative", 1), 
        (1, 1, "positive", 2),  
        (1, 1, "negative", 2),  
    ]
    
    for base_idx, slope_idx, reflection_type, target_idx in collision_cases:
        if len(coordinates) <= target_idx or len(slopes) <= slope_idx:
            continue
            
        base_x, base_y = coordinates[base_idx]
        target_x, target_y = coordinates[target_idx]
        base_slope = slopes[slope_idx]
        
        intersection_x, intersection_y = calculate_reflection_path(
            base_slope, base_x, base_y, reflection_type, target_x, target_y
        )
        
        # 检查后续路径是否与板凳发生碰撞
        for i, (current_x, current_y) in enumerate(coordinates[:-1]):
            if len(theta_positions) > i+1 and theta_positions[i+1] - start_theta >= np.pi:
                current_slope = slopes[i]
                
                # 点到直线距离公式
                distance_to_bench = (abs(current_slope*(intersection_x - current_x) + current_y - intersection_y) / 
                                   np.sqrt(current_slope**2 + 1))
                
                if distance_to_bench < BENCH_RADIUS:
                    return True
    
    return False

def evaluate_pitch_fitness(pitch: float, turning_diameter: float) -> Tuple[bool, float]:
    """
    评估螺距的适应度
    返回: (是否发生碰撞, 适应度分数)
    适应度分数：无碰撞时分数更高，螺距越小分数越高
    """
    min_theta = turning_diameter * np.pi / pitch
    
    # 测试多个极角位置
    collision_count = 0
    total_tests = 0
    
    for theta in np.arange(min_theta + 6, min_theta, -0.1):
        total_tests += 1
        # 计算龙身轨迹
        theta_positions = calculate_dragon_positions(theta, pitch)
        coordinates = convert_to_cartesian(theta_positions, pitch)
        slopes = calculate_path_slopes(coordinates)
        
        # 检测碰撞
        if check_collision_with_benches(theta, coordinates, slopes, theta_positions):
            collision_count += 1
    
    # 计算适应度：无碰撞 + 螺距小 = 高适应度
    has_collision = collision_count > 0
    if has_collision:
        # 有碰撞：适应度与碰撞率成反比，与螺距大小成正比
        fitness = (1 - collision_count / total_tests) * 100 + pitch * 10
    else:
        # 无碰撞：适应度与螺距成反比（越小越好）
        fitness = 1000 / pitch
    
    return has_collision, fitness

class GeneticAlgorithm:
    """遗传算法优化器"""
    
    def __init__(self, 
                 population_size: int = 50,
                 elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def initialize_population(self, pitch_min: float, pitch_max: float) -> List[float]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            pitch = random.uniform(pitch_min, pitch_max)
            population.append(pitch)
        return population
    
    def selection(self, population: List[float], fitness_scores: List[float]) -> List[float]:
        """选择操作 - 锦标赛选择"""
        selected = []
        
        # 精英保留
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序排列
        for i in range(self.elite_size):
            selected.append(population[sorted_indices[i]])
        
        # 锦标赛选择填充剩余位置
        while len(selected) < self.population_size:
            # 随机选择3个个体进行锦标赛
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, parent1: float, parent2: float) -> Tuple[float, float]:
        """交叉操作 - 算术交叉"""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate(self, individual: float, pitch_min: float, pitch_max: float) -> float:
        """变异操作 - 高斯变异"""
        if random.random() < self.mutation_rate:
            # 高斯变异，标准差为搜索范围的5%
            sigma = (pitch_max - pitch_min) * 0.05
            mutated = individual + random.gauss(0, sigma)
            # 确保在有效范围内
            mutated = max(pitch_min, min(pitch_max, mutated))
            return mutated
        return individual
    
    def evolve_generation(self, population: List[float], fitness_scores: List[float], 
                         pitch_min: float, pitch_max: float) -> List[float]:
        """进化一代"""
        # 选择
        selected = self.selection(population, fitness_scores)
        
        # 交叉和变异
        new_population = []
        
        # 保留精英
        for i in range(self.elite_size):
            new_population.append(selected[i])
        
        # 生成新个体
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1, pitch_min, pitch_max)
            child2 = self.mutate(child2, pitch_min, pitch_max)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]

def genetic_optimize_spiral_pitch(initial_velocity=1.0, turning_diameter=9.0, 
                                 max_generations=100, target_precision=1e-6):
    """
    使用遗传算法优化螺距
    """
    print("=" * 60)
    print("舞龙调头空间螺距遗传算法优化开始")
    print(f"初始参数: 龙头速度={initial_velocity} m/s, 调头空间直径={turning_diameter} m")
    print(f"遗传算法参数: 种群大小=50, 最大代数={max_generations}")
    print("=" * 60)
    
    # 搜索范围
    pitch_min, pitch_max = 0.4, 0.6
    
    # 初始化遗传算法
    ga = GeneticAlgorithm(population_size=50, elite_ratio=0.2, 
                          mutation_rate=0.1, crossover_rate=0.8)
    
    # 初始化种群
    population = ga.initialize_population(pitch_min, pitch_max)
    
    best_pitch = None
    best_fitness = -float('inf')
    no_improvement_count = 0
    
    # 全局记录所有发现的碰撞螺距和无碰撞螺距
    all_collision_pitches = []
    all_no_collision_pitches = []
    
    for generation in range(max_generations):
        # 评估种群适应度
        fitness_scores = []
        collision_pitches = []
        no_collision_pitches = []
        
        for pitch in population:
            has_collision, fitness = evaluate_pitch_fitness(pitch, turning_diameter)
            fitness_scores.append(fitness)
            if has_collision:
                collision_pitches.append(pitch)
                all_collision_pitches.append(pitch)  # 记录到全局列表
            else:
                no_collision_pitches.append(pitch)
                all_no_collision_pitches.append(pitch)  # 记录无碰撞螺距
        
        # 更新最佳解
        current_best_idx = np.argmax(fitness_scores)
        current_best_pitch = population[current_best_idx]
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_pitch = current_best_pitch
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 输出进度
        if generation % 10 == 0 or generation < 10:
            collision_count = len(collision_pitches)
            no_collision_count = len(no_collision_pitches)
            min_no_collision_pitch = min(no_collision_pitches) if no_collision_pitches else "无"
            global_min_no_collision = min(all_no_collision_pitches) if all_no_collision_pitches else "无"
            print(f"第{generation:3d}代: 最佳螺距={current_best_pitch:.6f}, "
                  f"适应度={current_best_fitness:.2f}, 碰撞个体={collision_count}/50, 无碰撞个体={no_collision_count}/50, "
                  f"本代最小无碰撞螺距={min_no_collision_pitch if isinstance(min_no_collision_pitch, str) else f'{min_no_collision_pitch:.6f}'}, "
                  f"全局最小无碰撞螺距={global_min_no_collision if isinstance(global_min_no_collision, str) else f'{global_min_no_collision:.6f}'}")
        
        # 早停条件
        if no_improvement_count >= 20:
            print(f"连续20代无改进，提前终止于第{generation}代")
            break
        
        # 进化到下一代
        population = ga.evolve_generation(population, fitness_scores, pitch_min, pitch_max)
    
    # 目标：找到最小的无碰撞螺距
    if all_no_collision_pitches:
        optimal_pitch = min(all_no_collision_pitches)
        print(f"\n从{len(all_no_collision_pitches)}个无碰撞螺距中找到最小值: {optimal_pitch:.6f}")
    elif all_collision_pitches:
        # 如果所有螺距都有碰撞，选择碰撞最少的（适应度最高的）
        optimal_pitch = min(all_collision_pitches)
        print(f"\n所有螺距都有碰撞，从{len(all_collision_pitches)}个碰撞螺距中选择最小值: {optimal_pitch:.6f}")
    else:
        # 兜底情况
        optimal_pitch = best_pitch if best_pitch is not None else 0.5
        print(f"\n未发现任何有效螺距，使用最佳适应度螺距: {optimal_pitch:.6f}")
    
    return optimal_pitch

# ========== 主程序执行 ==========
if __name__ == "__main__":
    try:
        # 设置随机种子以保证结果可重现
        random.seed(42)
        np.random.seed(42)
        
        # 开始遗传算法优化
        result = genetic_optimize_spiral_pitch(max_generations=50)
        
        print("\n" + "=" * 60)
        print("遗传算法优化结果")
        print("=" * 60)
        print(f"在调头空间外不会发生碰撞的最小螺距: {result:.7f} m")
        
        # 验证结果
        print(f"\n结果验证:")
        min_theta = 9.0 * np.pi / result
        print(f"   对应的最小极角: {min_theta:.4f} rad ({min_theta*180/np.pi:.2f}°)")
        print(f"   螺线紧密度: {2*np.pi/result:.4f} rad/m")
        
        # 最终验证
        has_collision, fitness = evaluate_pitch_fitness(result, 9.0)
        print(f"   最终验证: {'有碰撞' if has_collision else '无碰撞'}, 适应度={fitness:.2f}")
        
        print("\n优化完成!")
        
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        raise 