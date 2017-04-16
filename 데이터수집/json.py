# coding=utf-8
import json
import requests

serialized = """{"title" : "Data Science Book",
                "author" : "Joel Grus"
                "publicationYear" : 2014,
                "topics" : [ "data", "science", "data science"] }"""

# JSON을 파이썬 dict로 파싱
deserialized_json = json.loads(serialized)
if "data science" in deserialized_json["topics"]:
    print deserialized_json


endpoint = "https://api.github.com/users/joelgrus/repos"

repos = json.loads(requests.get(endpoint).text)

