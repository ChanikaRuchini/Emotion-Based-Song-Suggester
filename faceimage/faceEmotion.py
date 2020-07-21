import http.client, urllib.request, urllib.parse, urllib.error, base64,sys
import simplejson as json 
from tkinter import messagebox

#Detect emotion of the face image taken by the web cam using Microsoft Emotion API      
def checkEmotion():
        headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': 'ba4dc76507e34d4e87d6fb1325afb128',
	}

        params = urllib.parse.urlencode({
        # Request parameters
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'emotion'
         })

        try:
        	conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
        	data = open(r'images/frame.png', 'rb')
        	conn.request("POST", "/face/v1.0/detect?%s" % params,data, headers)
        	response = conn.getresponse()
        	data = response.read()
        	print(data)
        	parsed = json.loads(data)         	
        	if len(parsed)==0:
        		res = " "
        	else :
        		val = parsed[0]["faceAttributes"]["emotion"]
        		res = max(val, key = val.get) 
        		print ("\nEmotion :: ",res) 
        	conn.close()
        except Exception as e:
        	print("[Errno {0}] {1}".format(e.errno, e.strerror))
        	list1="[Errno {0}] {1}".format(e.errno, e.strerror)
        	messagebox.showerror('No internet connection', 'Please check again.')
        return res

