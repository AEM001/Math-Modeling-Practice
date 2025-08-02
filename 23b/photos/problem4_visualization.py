import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Set font to a locally available one that supports Chinese characters
mpl.rcParams['font.sans-serif'] = ['Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False

INPUT_FILE = "附件.xlsx"
NM_TO_M = 1852  # Conversion factor for nautical miles to meters

# --- Data Loading and Preparation ---
def load_and_prepare_data(file_path):
    """
    Loads and prepares the depth data from the Excel file by manually parsing
    the specific matrix/pivot table format, bypassing pandas' header logic.
    """
    try:
        # Read the entire sheet without any header or index processing
        df = pd.read_excel(file_path, header=None)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。")
        return None, None, None
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return None, None, None

    # Manually extract coordinates and data
    # X coordinates are in the second row (index 1), starting from the second column (index 1)
    x_coords_series = pd.to_numeric(df.iloc[1, 2:], errors='coerce')
    x_coords = x_coords_series.dropna().to_numpy()
    
    # Y coordinates are in the first column (index 0), starting from the third row (index 2)
    y_coords_series = pd.to_numeric(df.iloc[2:, 1], errors='coerce')
    y_coords = y_coords_series.dropna().to_numpy()
    
    # Depth values are in the main body of the table, starting at cell C3 (row 2, col 2)
    depth_values = df.iloc[2:, 2:].apply(pd.to_numeric, errors='coerce').values

    # Create a meshgrid for broadcasting coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten the arrays to create the long-format data
    x = xx.flatten()
    y = yy.flatten()
    z = depth_values.flatten()

    # Create a final DataFrame and drop any rows with missing data
    final_df = pd.DataFrame({'x_nm': x, 'y_nm': y, 'z_m': z}).dropna()

    if final_df.empty:
        print("错误: 解析后数据为空，请检查Excel文件内容和脚本中的索引。")
        return None, None, None

    # Convert nautical miles to meters and extract data.
    x_out = final_df['x_nm'].to_numpy() * NM_TO_M
    y_out = final_df['y_nm'].to_numpy() * NM_TO_M
    z_out = final_df['z_m'].to_numpy()
    
    return x_out, y_out, z_out

# --- Visualization Functions ---

def plot_scatter(x, y, z):
    """Creates and saves a 2D scatter plot with depth represented by color."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=z, cmap='viridis_r', s=5)
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('海水深度 (m)')
    
    ax.set_xlabel('东西方向距离 (m)')
    ax.set_ylabel('南北方向距离 (m)')
    ax.set_title('2D 深度数据散点图', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("depth_scatter_plot.png", dpi=300)
    print("2D 深度散点图已保存至: depth_scatter_plot.png")
    plt.close(fig)

def plot_contour(x, y, z):
    """Creates and saves a 2D filled contour plot by interpolating the data."""
    # Create a grid to interpolate onto
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    
    # Interpolate the scattered data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis_r', levels=40)
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('海水深度 (m)')
    
    ax.set_xlabel('东西方向距离 (m)')
    ax.set_ylabel('南北方向距离 (m)')
    ax.set_title('2D 深度数据等高线图', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("depth_contour_plot.png", dpi=300)
    print("2D 等高线图已保存至: depth_contour_plot.png")
    plt.close(fig)

def plot_3d_surface(x, y, z):
    """Creates and saves a 3D surface plot by interpolating the data."""
    # Create a grid for interpolation
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # The plot_surface command requires the X, Y, Z coordinates to be 2D arrays
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis_r', edgecolor='none')
    
    # Invert Z-axis so that deeper values are lower
    ax.set_zlim(np.nanmax(grid_z), np.nanmin(grid_z)) 
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='海水深度 (m)')
    
    ax.set_xlabel('东西方向 (m)')
    ax.set_ylabel('南北方向 (m)')
    ax.set_zlabel('海水深度 (m)')
    ax.set_title('3D 海底地形曲面图', fontsize=16)
    
    # Adjust viewing angle
    ax.view_init(elev=45, azim=-120)
    
    plt.tight_layout()
    plt.savefig("depth_3d_surface_plot.png", dpi=300)
    print("3D 海底地形曲面图已保存至: depth_3d_surface_plot.png")
    plt.close(fig)

def plot_contour_with_labels(x, y, z):
    """Creates and saves a 2D contour plot with inline labels."""
    # Create a grid to interpolate onto
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    
    # Interpolate the scattered data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define number of levels for contour lines
    num_levels = 30

    # Draw filled contours for color background
    contour_filled = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis_r', levels=num_levels, alpha=0.9)
    
    # Add a color bar for the filled contour
    cbar = fig.colorbar(contour_filled, ax=ax, shrink=0.8)
    cbar.set_label('海水深度 (m)')

    # Draw contour lines over the filled contours for labeling
    contour_lines = ax.contour(grid_x, grid_y, grid_z, colors='black', levels=contour_filled.levels, linewidths=0.7)
    
    # Add labels to the contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%d m')
    
    ax.set_xlabel('东西方向距离 (m)')
    ax.set_ylabel('南北方向距离 (m)')
    ax.set_title('带标签的2D深度等高线图', fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("depth_contour_plot_with_labels.png", dpi=300)
    print("带标签的2D等高线图已保存至: depth_contour_plot_with_labels.png")
    plt.close(fig)

# --- Main Execution ---
if __name__ == '__main__':
    x_data, y_data, z_data = load_and_prepare_data(INPUT_FILE)
    
    if x_data is not None:
        plot_scatter(x_data, y_data, z_data)
        plot_contour(x_data, y_data, z_data)
        plot_3d_surface(x_data, y_data, z_data)
        plot_contour_with_labels(x_data, y_data, z_data)
        print("\n所有可视化图表已生成完毕。") 