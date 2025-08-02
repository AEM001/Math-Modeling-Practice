import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# Set font to a locally available one that supports Chinese characters
mpl.rcParams['font.sans-serif'] = ['Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False

def design_survey_lines():
    """
    Calculates the optimal survey line placement for Problem 3 and generates results.
    """
    # 1. Constants and Parameters
    D_0 = 110  # Depth at center (m)
    alpha_deg = 1.5  # Slope (degrees)
    theta_deg = 120  # Transducer open angle (degrees)
    eta = 0.10  # Target overlap ratio (10%)
    L_ew_nm = 4  # East-West length (nautical miles)
    L_ns_nm = 2  # North-South length (nautical miles)
    NM_TO_M = 1852  # Conversion factor for nautical miles to meters

    L_ew_m = L_ew_nm * NM_TO_M
    L_ns_m = L_ns_nm * NM_TO_M
    
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)

    # 2. Coordinate System & Initial Calculations
    # x=0 at the Western edge, positive towards East.
    # Depth at center (x = L_ew_m / 2) is D_0.
    # The depth at the Western edge (x=0) is the maximum depth.
    D_max = D_0 + (L_ew_m / 2) * np.tan(alpha)

    # 3. Iterative Line Placement
    # The position of the first line is calculated to cover the western boundary (x=0).
    x_1 = D_max * np.tan(theta / 2)
    
    lines_positions = [x_1]
    x_k = x_1
    
    while True:
        # Depth at the current line position
        D_k = D_max - x_k * np.tan(alpha)
        
        # Check if the coverage of the current line already spans the entire area.
        W_k_right_proj = (D_k * np.sin(theta / 2) / np.cos(theta / 2 - alpha)) * np.cos(alpha)
        if (x_k + W_k_right_proj) >= L_ew_m:
            break

        # Calculate the distance d_k to the next survey line using the derived formula
        W_k_left = D_k * np.sin(theta / 2) / np.cos(theta / 2 + alpha)
        W_k_right = D_k * np.sin(theta / 2) / np.cos(theta / 2 - alpha)
        
        numerator = (W_k_left + W_k_right) * (1 - eta) * np.cos(alpha)
        denominator = 1 + (1 - eta) * np.sin(alpha) * np.sin(theta / 2) / np.cos(theta / 2 + alpha)
        d_k = numerator / denominator
        
        # Position of the next line
        x_k_plus_1 = x_k + d_k
        lines_positions.append(x_k_plus_1)
        
        x_k = x_k_plus_1

    # 4. Results Processing and Output
    results_data = []
    for i, pos in enumerate(lines_positions):
        depth = D_max - pos * np.tan(alpha)
        W_left_proj = (depth * np.sin(theta / 2) / np.cos(theta / 2 + alpha)) * np.cos(alpha)
        W_right_proj = (depth * np.sin(theta / 2) / np.cos(theta / 2 - alpha)) * np.cos(alpha)
        
        left_edge = pos - W_left_proj
        right_edge = pos + W_right_proj
        
        dist_to_next = lines_positions[i+1] - pos if i < len(lines_positions) - 1 else np.nan
        
        # Calculate actual overlap with the previous line
        if i > 0:
            prev_pos = lines_positions[i-1]
            prev_depth = D_max - prev_pos * np.tan(alpha)
            prev_W_right_proj = (prev_depth * np.sin(theta/2) / np.cos(theta/2 - alpha)) * np.cos(alpha)
            prev_right_edge = prev_pos + prev_W_right_proj
            
            overlap_dist = prev_right_edge - left_edge
            
            # Use the IHO definition for overlap rate denominator: distance between outer edges of two adjacent swaths
            denominator_w = right_edge - prev_pos + prev_W_right_proj
            # A simpler way is to use swath width of current line
            total_width = W_left_proj + W_right_proj
            overlap_perc = (overlap_dist / total_width) * 100 if total_width > 0 else 0
        else:
            overlap_perc = np.nan

        results_data.append([i + 1, pos, depth, dist_to_next, left_edge, right_edge, overlap_perc])

    df = pd.DataFrame(results_data, columns=[
        '测线号', '离西侧距离 (m)', '水深 (m)', 
        '与下一条测线间距 (m)', '左覆盖边界 (m)', '右覆盖边界 (m)', 
        '与上一条重叠率 (%)'
    ])

    total_lines = len(df)
    total_length_km = (total_lines * L_ns_m) / 1000

    print("="*20 + " 测线设计结果 " + "="*20)
    print(f"海域东西宽度: {L_ew_m:.2f} m")
    print(f"所需测线总数: {total_lines}")
    print(f"测线总长度: {total_length_km:.2f} km")
    print("\n详细测线参数:")
    print(df.to_string())
    
    # Save results to file
    output_path = "3/result3.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n结果已保存至 {output_path}")
    
    return df, D_max, L_ew_m, alpha

def visualize_results(df, D_max, L_ew_m, alpha):
    """
    Generates and saves plots for the survey design.
    """
    # 1. Plot Seabed Profile
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x_coords = np.linspace(0, L_ew_m, 500)
    y_coords_depth = D_max - x_coords * np.tan(alpha)
    ax1.plot(x_coords, y_coords_depth, label='海底剖面')
    ax1.invert_yaxis()  # Depth increases downwards
    ax1.set_xlabel('离西侧距离 (m)')
    ax1.set_ylabel('水深 (m)')
    ax1.set_title('待测海域东西向海底剖面图')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    plt.tight_layout()
    plt.savefig("3/seabed_profile.png")
    print("海底剖面图已保存至 3/seabed_profile.png")

    # 2. Plot Coverage Diagram
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    
    # Plot sea area boundaries
    ax2.axvline(0, color='k', linestyle='--', label='海域西边界')
    ax2.axvline(L_ew_m, color='k', linestyle='--', label='海域东边界')

    for i, row in df.iterrows():
        line_pos = row['离西侧距离 (m)']
        left_edge = row['左覆盖边界 (m)']
        right_edge = row['右覆盖边界 (m)']
        width = right_edge - left_edge
        
        # Draw the main swath
        rect = patches.Rectangle((left_edge, i + 0.5), width, 0.5, 
                                 edgecolor='black', facecolor='skyblue', alpha=0.6)
        ax2.add_patch(rect)
        # Mark the survey line
        ax2.plot([line_pos, line_pos], [i + 0.5, i + 1.0], color='red', lw=2)

    ax2.set_yticks(np.arange(len(df)) + 0.75)
    ax2.set_yticklabels(df['测线号'])
    ax2.set_xlabel('离西侧距离 (m)')
    ax2.set_ylabel('测线号')
    ax2.set_title('测线布设与覆盖范围示意图')
    ax2.set_xlim(-500, L_ew_m + 500)
    ax2.set_ylim(0, len(df) + 1)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("3/survey_coverage_diagram.png")
    print("测线覆盖示意图已保存至 3/survey_coverage_diagram.png")


if __name__ == '__main__':
    results_df, D_max, L_ew_m, alpha = design_survey_lines()
    visualize_results(results_df, D_max, L_ew_m, alpha) 