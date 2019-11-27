import requests
def send_line_notification(message):
    line_token = '7T12MMobxZMTOMrvNOMgFnfuVSnj2qxaj4oT9SW4iHT' # 終わったら無効化する
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)
    
