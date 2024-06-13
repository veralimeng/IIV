import requests

def handler(event, context):
    url = "https://raw.githubusercontent.com/veralimeng/IIV/main/code.py"
    
    response = requests.get(url)
    with open("/tmp/code.py", "w") as f:
        f.write(response.text)
    
    exec(open("/tmp/code.py").read())
    
    return {
        "statusCode": 200,
        "body": "code.py executed successfully"
    }
