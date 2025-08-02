import pandas as pd
import numpy as np

try:
    from skbio.stats.composition import ilr
except ImportError:
    print("scikit-bio is not installed. Please install it using: pip install scikit-bio")
    # In a library, it's better to raise the error than to exit.
    raise

def ilr_transform(chem_data_df: pd.DataFrame) -> np.ndarray:
    """
    Performs Isometric Log-Ratio (ILR) transformation on chemical composition data.
    It replaces zeros with a constant value (65% of the minimum non-zero value) 
    before applying the transformation.

    Args:
        chem_data_df: A pandas DataFrame where each row is a sample and each 
                      column is a chemical component.

    Returns:
        A NumPy array containing the ILR transformed coordinates.
    """
    if not isinstance(chem_data_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    chem_data = chem_data_df.astype(float).values
    
    # Create a copy to avoid modifying the original data
    chem_data_processed = chem_data.copy()

    # Handle the case where all data is zero
    if np.all(chem_data_processed == 0):
        return np.zeros((chem_data_processed.shape[0], chem_data_processed.shape[1] - 1))

    # Find the global minimum non-zero value
    if np.any(chem_data_processed > 0):
        min_nonzero = np.min(chem_data_processed[chem_data_processed > 0])
        # Use 65% of the minimum non-zero value as the replacement for zeros
        replacement_val = min_nonzero * 0.65
        chem_data_processed[chem_data_processed == 0] = replacement_val
    
    # skbio.ilr automatically handles closure (scaling rows to sum to 1)
    ilr_results = ilr(chem_data_processed)
    
    return ilr_results

# --- Main execution block for standalone script usage ---
def main():
    """
    Main function to run the ILR transformation as a standalone script.
    Loads data, transforms it, and saves the result. Assumes execution from project root.
    """
    # --- 1. 加载数据 ---
    input_path = '2/2.1/附件2_处理前.csv'
    output_path = '2/2.1/附件2_处理后_ILR_常数替换.csv'
    
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Make sure you are running this script from the project root directory.")
        return
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='gbk')

    print(f"数据已从 {input_path} 加载。")

    # --- 2. 准备数据 ---
    try:
        first_chem_col_index = df.columns.get_loc('二氧化硅(SiO2)')
    except KeyError:
        print("Error: '二氧化硅(SiO2)' column not found. Cannot identify chemical composition columns.")
        return
        
    chem_cols = df.columns[first_chem_col_index:]
    info_cols = df.columns[:first_chem_col_index]
    chem_data_df = pd.DataFrame(df[chem_cols])

    print("化学成分数据已提取。")

    # --- 3. ILR变换 ---
    print("开始执行零替换和ILR变换...")
    ilr_results = ilr_transform(chem_data_df)

    # 创建ILR结果的DataFrame
    ilr_col_names = list(chem_cols[:-1])
    ilr_df = pd.DataFrame(ilr_results, columns=ilr_col_names, index=df.index)

    # 合并信息列和ILR结果
    final_df = pd.concat([df[info_cols], ilr_df], axis=1)
    print("ILR变换完成。")


    # --- 4. 保存结果 ---
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"处理完成，结果已保存到 {output_path}")
    print("\n处理后的数据预览：")
    print(final_df.head())

if __name__ == '__main__':
    main() 