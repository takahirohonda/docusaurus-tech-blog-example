---
slug: data-engineering/data-ingestion/python-xml-to-json
title: Sending XML Payload and Converting XML Response to JSON with Python
tags: [Data Engineering, Data Ingestion, Python, QAS, XML]
---

If you need to interact with a REST endpoint that takes a XML string as a payload and returns another XML string as a response, this is the quick guide if you want to use Python.<!-- truncate -->

If you want to do it with Node.js, you can check out the post here.

Just like the example with Node.js, we will use QAS endpoint (see more info on this in the intro of the node.js version post).

The idea is very simple. We will pass the XML payload as a string and convert the response XML string to a dictionary by using xmltodict and parse it to JSON. For an API call, we use the requests module.

```python
import requests, json, xmltodict

request_url = 'https://ws.ondemand.qas.com/ProOnDemand/V3/ProOnDemandService.asmx?WSDL='
qas_token = '<add your token>'
soap_action = 'http://www.qas.com/OnDemand-2011-03/DoSearch'
headers = {
  'Content-Type':'text/xml',
  'Auth-Token':qas_token,
  'SOAPAction': soap_action
  }

xml = '<?xml version="1.0" encoding="UTF-8" ?>\
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:ond="http://www.qas.com/OnDemand-2011-03">\
  <soapenv:Header>\
    <ond:QAQueryHeader />\
  </soapenv:Header>\
  <soapenv:Body>\
    <ond:QASearch>\
      <ond:Country>AUS</ond:Country>\
      <ond:Engine Flatten="true" Intensity="Close" PromptSet="Default" Threshold="5" Timeout="10000">Intuitive</ond:Engine>\
      <ond:Layout></ond:Layout>\
      <ond:Search>101 Colll</ond:Search>\
    </ond:QASearch>\
  </soapenv:Body>\
</soapenv:Envelope>'


r = requests.post(request_url, data=xml, headers=headers)
json_results = json.dumps(xmltodict.parse(r.text))
print(json_results)
```

That’s it. Pretty easy!
