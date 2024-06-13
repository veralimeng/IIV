import os
import requests
import zipfile

def handler(event, context):
    zip_url = "https://github.com/veralimeng/IIV/archive/refs/heads/main.zip"
    zip_path = "/tmp/main.zip"
    extract_path = "/tmp/main"

    try:
        print(f"Starting download from {zip_url}")
        response = requests.get(zip_url)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print(f"ZIP file downloaded successfully to {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"ZIP file extracted successfully to {extract_path}")

        code_file_path = os.path.join(extract_path, "IIV-main", "code.py")
        print(f"Code file path: {code_file_path}")

        if os.path.exists(code_file_path):
            print("Executing code.py")
            exec(open(code_file_path).read())
            print("Code executed successfully.")
        else:
            raise FileNotFoundError(f"code.py not found at {code_file_path}")

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
    except FileNotFoundError as e:
        error_message = f"File not found: {e}"
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
