import os
import requests
import zipfile

def handler(event, context):
    zip_url = "https://github.com/veralimeng/IIV/archive/refs/heads/main.zip"
    zip_path = "/tmp/main.zip"
    extract_path = "/tmp/main"

    try:
        # 下载 ZIP 文件
        response = requests.get(zip_url)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("ZIP file downloaded successfully.")

        # 解压 ZIP 文件
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("ZIP file extracted successfully.")

        # 确定解压后文件路径
        code_file_path = os.path.join(extract_path, "IIV-main", "code.py")
        print(f"Code file path: {code_file_path}")

        # 执行解压后的 code.py 文件
        exec(open(code_file_path).read())
        print("Code file executed successfully.")
        
        return {
            "statusCode": 200,
            "body": "code.py executed and run successfully"
        }
    except requests.exceptions.RequestException as e:
        error_message = f"Failed to download code.py: {e}"
        print(error_message)
        return {
            "statusCode": 500,
            "body": error_message
        }
    except zipfile.BadZipFile as e:
        error_message = f"Failed to unzip the file: {e}"
        print(error_message)
        return {
            "statusCode": 500,
            "body": error_message
        }
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return {
            "statusCode": 500,
            "body": error_message
        }
