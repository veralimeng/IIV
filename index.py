import requests
import os

def handler(event, context):
    # 下载 code.py 文件
    code_url = "https://raw.githubusercontent.com/veralimeng/IIV/main/code.py"
    # data_url1 = "https://raw.githubusercontent.com/Vera-lm/IIV/main/crunchbase_export-4-10-2019.xlsx"
    # data_url2 = "https://raw.githubusercontent.com/Vera-lm/IIV/main/combined_tweet_df.csv"
    # data_url3 = "https://raw.githubusercontent.com/Vera-lm/IIV/main/user.csv"

    try:
        # 下载 code.py 文件
        code_response = requests.get(code_url, timeout=10)
        code_response.raise_for_status()
        code_path = "/tmp/code.py"
        with open(code_path, "w") as f:
            f.write(code_response.text)

        # # 下载数据文件
        # data_response1 = requests.get(data_url1, timeout=10)
        # data_response1.raise_for_status()
        # data_path1 = "/tmp/crunchbase_export-4-10-2019.xlsx"
        # with open(data_path1, "wb") as f:
        #     f.write(data_response1.content)
        
        # data_response2 = requests.get(data_url2, timeout=10)
        # data_response2.raise_for_status()
        # data_path2 = "/tmp/combined_tweet_df.csv"
        # with open(data_path2, "wb") as f:
        #     f.write(data_response2.content)
        
        # data_response3 = requests.get(data_url3, timeout=10)
        # data_response3.raise_for_status()
        # data_path3 = "/tmp/user.csv"
        # with open(data_path3, "wb") as f:
        #     f.write(data_response3.content)

        # 打印 code.py 的内容进行调试
        with open(code_path, "r") as f:
            code_content = f.read()
        print(f"code.py 内容:\n{code_content}")

        # 执行 code.py
        exec(code_content)

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
