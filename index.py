import requests
import os
import sys
import time

def download_with_retries(url, dest_path, retries=5, timeout=120):
    attempt = 0
    while attempt < retries:
        try:
            print(f"尝试下载文件: {url}, 尝试次数: {attempt + 1}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            print(f"文件下载成功: {url}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"下载错误: {str(e)}, 尝试次数: {attempt + 1}")
            attempt += 1
            time.sleep(5)  # 等待 5 秒后重试
    return False

def handler(event, context):
    # 打印 Python 版本
    print("Python 版本:", sys.version)

    # 下载文件
    code_url = "https://raw.githubusercontent.com/veralimeng/IIV/main/code.py"
    data_url1 = "https://github.com/veralimeng/IIV/raw/main/funding_twitter_df_copy2.csv.zip"

    code_path = "/tmp/code.py"
    data_path1 = "/tmp/funding_twitter_df_copy2.csv.zip"

    if not download_with_retries(code_url, code_path):
        return {
            "statusCode": 500,
            "body": "Failed to download code.py after multiple attempts"
        }

    if not download_with_retries(data_url1, data_path1):
        return {
            "statusCode": 500,
            "body": "Failed to download data file after multiple attempts"
        }

    # 打印 code.py 的内容进行调试
    with open(code_path, "r") as f:
        code_content = f.read()
    print(f"code.py 前几行内容:\n{code_content[:500]}")  # 打印前500个字符

    # 检查 code.py 内容的语法
    try:
        code = compile(code_content, code_path, 'exec')
    except SyntaxError as e:
        print(f"编译 code.py 时发生语法错误: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Syntax Error in code.py: {str(e)}"
        }

    print("开始执行 code.py 文件...")
    # 执行 code.py
    try:
        exec(code)
        print("code.py 文件执行完成")
        return {
            "statusCode": 200,
            "body": "code.py executed successfully"
        }
    except Exception as e:
        print(f"执行错误: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Execution Error: {str(e)}"
        }
