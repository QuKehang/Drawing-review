import pandas as pd
import re

# 读取文本文件
def read_text_file(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 读取 Excel 文件
def read_excel_file(excel_file_path):
    df = pd.read_excel(excel_file_path)
    df['累计张数'] = df['张数'].cumsum()
    return df

# 提取中文字符
def extract_chinese(text):
    return ''.join(re.findall(r'[\u4e00-\u9fff]', text))

# 提取数字
def extract_numbers(text):
    return ''.join(re.findall(r'\d+', text))

# 处理图号中的非数字字符
def clean_number(text):
    return re.sub(r'\D', '', text)

# 提取特定信息
def extract_info_from_line(line, line_number):
    if line_number == 1:
        page_match = re.search(r'page_(\d+)', line)
        return page_match.group(1) if page_match else '未找到页码'
    
    elif line_number == 2:
        box1_match = re.search(r'box1:\s*([^|]+)', line)
        box1_content = box1_match.group(1).strip() if box1_match else '未找到 box1 内容'
        return extract_chinese(box1_content)
    
    elif line_number == 3:
        box2_match = re.search(r'box2:\s*([^~]+)', line)
        box2_content = box2_match.group(1).strip() if box2_match else '未找到 box2 内容'
        return extract_numbers(box2_content)

    return '未处理的行'

# 对比图号
def compare_numbers(txt_number, excel_number):
    txt_number = clean_number(txt_number)
    excel_number = clean_number(excel_number)

    if len(excel_number) == 5:
        return txt_number[:5] == excel_number
    elif len(excel_number) == 8:
        part1, part2, part3 = excel_number[:2], excel_number[2:5], excel_number[5:]
        if len(txt_number) == 5:
            txt_part1 = txt_number[:2]
            txt_part2 = txt_number[2:]
            if txt_part1 == part1:
                return part2 <= txt_part2 <= part3
    return False

# 放宽图名匹配条件
def is_substring(sub, main):
    return all(char in main for char in sub)

# 对比数据
def compare_with_excel(page_num, extracted_图名, extracted_图号, df):
    corresponding_row = None
    for i in range(len(df)):
        if page_num <= df.loc[i, '累计张数']:
            corresponding_row = df.iloc[i]
            break

    if corresponding_row is not None:
        excel_图名 = corresponding_row['图名']
        excel_图号 = corresponding_row['图号']
        图名_match = is_substring(extracted_图名, excel_图名)
        图号_match = compare_numbers(extracted_图号, excel_图号)
        return corresponding_row, 图名_match, 图号_match

    return None, False, False

# 处理每一行并进行对比
def process_lines(text_lines, df):
    results = []
    page_num = None
    extracted_图名 = None

    for idx, line in enumerate(text_lines):
        line = line.strip()
        if line:
            extracted_info = extract_info_from_line(line, (idx % 4) + 1)  

            if (idx % 4) + 1 == 1:
                page_num = extracted_info
            elif (idx % 4) + 1 == 2:
                extracted_图名 = extracted_info
            elif (idx % 4) + 1 == 3:
                extracted_图号 = extracted_info
                corresponding_row, 图名_match, 图号_match = compare_with_excel(int(page_num), extracted_图名, extracted_图号, df)
                results.append({
                    'page_num': page_num,
                    'extracted_图名': extracted_图名,
                    'extracted_图号': extracted_图号,
                    '图名': corresponding_row['图名'] if corresponding_row is not None else '无对应行',
                    '图号': corresponding_row['图号'] if corresponding_row is not None else '无对应行',
                    '图名_match': 图名_match,
                    '图号_match': 图号_match
                })
            elif (idx % 4) + 1 == 4:  # 第四行不做处理
                continue

    return results

# 输出结果
def print_results(results):
    for result in results:
        print(f'页码: {result["page_num"]}')
        print(f'提取的图名: {result["extracted_图名"]}, 图名匹配: {result["图名_match"]}')
        print(f'提取的图号: {result["extracted_图号"]}, 图号匹配: {result["图号_match"]}')
        print('---')

# 主函数
def main(text_file_path, excel_file_path):
    text_lines = read_text_file(text_file_path)
    df = read_excel_file(excel_file_path)
    results = process_lines(text_lines, df)
    return results
