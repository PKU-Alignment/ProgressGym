import pandas as pd


def combine_csv_with_index(file1, file2, output_file, pref):
    # 读取两个CSV文件
    df1 = pd.read_csv(file1, on_bad_lines="skip")
    df2 = pd.read_csv(file2, on_bad_lines="skip")

    # 合并两个数据框
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 将原来的第一列改为顺次编号
    first_col = [pref + str(i) for i in range(1, 1 + len(combined_df))]
    combined_df.iloc[:, 0] = first_col

    # 将合并后的数据框写入输出文件
    combined_df.to_csv(output_file, index=False)


input_path = "src/evaluation/raw_dataset/"

combine_csv_with_index(
    input_path + "foundation" + "/generated.csv",
    input_path + "foundation" + "/prototype.csv",
    input_path + "foundation" + "/prototype.csv",
    "V_",
)
