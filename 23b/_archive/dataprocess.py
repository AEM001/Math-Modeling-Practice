import pandas as pd
import numpy as np
import openpyxl

def extract_excel_to_csv(excel_file_path, output_csv_path):
    """
    从Excel文件提取数据并转换为三列CSV格式：横坐标、纵坐标、深度
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file_path, header=None)
    
    # 提取横坐标 - 第2行（索引1）从第3列（索引2）开始
    horizontal_coords = df.iloc[1, 2:].values
    
    # 提取纵坐标 - 第2列（索引1）从第3行（索引2）开始
    vertical_coords = df.iloc[2:, 1].values
    
    # 提取数据矩阵 - 从第3行（索引2）第3列（索引2）开始
    data_matrix = df.iloc[2:, 2:].values

    # 为避免浮点精度问题，重新生成坐标
    # 我们知道步长是0.02，所以保留两位小数是合适的
    h_count = len(horizontal_coords)
    v_count = len(vertical_coords)
    
    # 使用 np.linspace 确保均匀间隔，然后四舍五入以校正精度
    h_coords_new = np.round(np.linspace(horizontal_coords[0], horizontal_coords[-1], h_count), 2)
    v_coords_new = np.round(np.linspace(vertical_coords[0], vertical_coords[-1], v_count), 2)
    
    # 创建结果列表
    result = []
    
    # 添加表头
    result.append(['横坐标', '纵坐标', '深度'])
    
    # 遍历数据矩阵
    for i, v_coord in enumerate(v_coords_new):
        for j, h_coord in enumerate(h_coords_new):
            depth_value = data_matrix[i, j]
            result.append([h_coord, v_coord, depth_value])
    
    # 转换为DataFrame并保存为CSV
    result_df = pd.DataFrame(result[1:], columns=result[0])
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"数据已成功提取并保存到: {output_csv_path}")
    print(f"总共提取了 {len(result_df)} 条数据记录")
    print(f"横坐标范围: {result_df['横坐标'].min()} - {result_df['横坐标'].max()}")
    print(f"纵坐标范围: {result_df['纵坐标'].min()} - {result_df['纵坐标'].max()}")
    
    # 显示前几行数据
    print("\n前10行数据预览:")
    print(result_df.head(10))
    
    return result_df


# 使用示例
if __name__ == "__main__":
    # 方法1: 如果你有Excel文件
    excel_file = "/Users/Mac/Downloads/23b/附件.xlsx"
    extract_excel_to_csv(excel_file, "output.csv")
    

