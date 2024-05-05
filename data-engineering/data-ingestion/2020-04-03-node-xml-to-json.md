---
slug: data-engineering/data-ingestion/Node-xml-to-json
title: Sending XML Payload and Converting XML Response to JSON with Node.js
tags: [Data Engineering, Data Ingestion, Node.js, QAS, XML]
---

Here is the quick Node.js example of interacting with a rest API endpoint that takes XML string as a payload and return with XML string as response. <!-- truncate -->Once we get the response, we will convert it to a JSON object.

For this example, we will use the old-school QAS (Quick Address Search). Although Experian moved to a more modern endpoint EDQ (Experian Data Quality), the endpoint is still available. Note that you need to sign up for a license. The purpose of this post is just to show how to interact with a rest API that uses XML. You can use this example for your own use case.

Firs of all, we use two dependencies, `axios` for making API calls and `xml2js` for converting xml string to JSON. Make sure that the xml string doesn’t contain any new line.

Then, you can pass xml as a string in the payload. When the response comes back as a xml string, you can convert it to JSON.

```js
const axios = require("axios");
const parseString = require("xml2js").parseString;

const qas_token = "<add your token>";
const soap_action = "http://www.qas.com/OnDemand-2011-03/DoSearch";
const xml =
  '<?xml version="1.0" encoding="UTF-8" ?> \
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
</soapenv:Envelope>';

const options = {
  method: "post",
  url: "https://ws.ondemand.qas.com/ProOnDemand/V3/ProOnDemandService.asmx?WSDL=",
  headers: {
    "Content-Type": "text/xml",

    "Auth-Token": qas_token,
    SOAPAction: soap_action,
  },
  data: xml,
};

axios(options)
  .then((response) => {
    console.log(response.data);
    parseString(response.data, (err, result) => {
      console.log(result);
      // console.log(JSON.stringify(result['soap:Envelope']['soap:Body']))
      var searchResult =
        result["soap:Envelope"]["soap:Body"][0]["QASearchResult"][0][
          "QAPicklist"
        ][0]["PicklistEntry"];
      searchResult.forEach((item) => {
        console.dir(item);
      });
    });
  })
  .catch((err) => {
    console.log(err);
  });
```
