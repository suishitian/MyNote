# -*- coding: utf-8 -*-
import requests
import md5
import random
import wave
import base64
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
asrUrl = "http://openapi.youdao.com/asrapi"
appKey = "0f656b89f991d548"
appSecret = "Qm2XyGslOwJMXKzGUHLz8VOd9T6cMqAo"

def asr(filename,lang,channel,rate,format):
    file_wav = open(filename, 'rb')
    q = base64.b64encode(file_wav.read())
    data = {}
    salt = random.randint(1, 65536)
    sign = appKey + q + str(salt) + appSecret
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()

    data['q'] = q
    data['langType'] = lang
    data['appKey'] = appKey
    data['salt'] = salt
    data['sign'] = sign
    data['format'] = format
    data['rate'] = rate
    data['channel'] = channel
    data['type'] = 1
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    ##print(data)
    response = requests.post(asrUrl,data=data,headers = headers)
    ##print(response.content)
    if response.json()["errorCode"]=="0":
        return response.json()["result"][0]
    else :
        return "error"

##filename = "/home/shitian/Downloads/usc_linux_voiceInput_sdk_v3.10.49/testfile/test1_16k.wav"

##response = asr(filename,"zh-CHS",1,16000,'wav')
##print(response)
