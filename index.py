import requests

def handler(event, context):
    url = "https://raw.githubusercontent.com/veralimeng/IIV/main/code.py"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("/tmp/code.py", "w") as f:
            f.write(response.text)
        
        exec(open("/tmp/code.py").read())
        
        return {
            "statusCode": 200,
            "body": "code.py executed successfully"
        }
    except requests.exceptions.RequestException as e:
        return {
            "statusCode": 500,
            "body": f"Failed to download or execute code.py: {e}"
        }
