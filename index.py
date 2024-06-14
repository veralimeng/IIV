import requests
import os

def handler(event, context):
    # 下载 code.py 文件
    code_url = "https://raw.githubusercontent.com/veralimeng/IIV/main/code.py"
    data_url1 = "https://github.com/veralimeng/IIV/raw/main/funding_twitter_df_copy2.csv.zip"

    try:
        # 下载 code.py 文件
        code_response = requests.get(code_url, timeout=60)
        code_response.raise_for_status()
        code_path = "/tmp/code.py"
        with open(code_path, "w") as f:
            f.write(code_response.text)

        # 下载数据文件
        data_response1 = requests.get(data_url1, timeout=60)
        data_response1.raise_for_status()
        data_path1 = "/tmp/funding_twitter_df_copy2.csv.zip"
        with open(data_path1, "wb") as f:
            f.write(data_response1.content)

        # 打印 code.py 的内容进行调试
        with open(code_path, "r") as f:
            code_content = f.read()
        print(f"code.py 内容:\n{code_content}")

        # 编译并执行 code.py
        code = compile(code_content, code_path, 'exec')
        exec(code)

        return {
            "statusCode": 200,
            "body": "code.py executed successfully"
        }
    except requests.exceptions.RequestException as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
    except SyntaxError as e:
        return {
            "statusCode": 500,
            "body": f"Syntax Error: {str(e)}"
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Execution Error: {str(e)}"
        }
