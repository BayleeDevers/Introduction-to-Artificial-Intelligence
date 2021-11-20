import requests
import json
import csv
import datetime
import time

exp_file = 'data/reddit_posts.csv'

try:
    with open(exp_file) as f:
        file_size = f.tell()
        f.seek(max(file_size, 0))
        # this will get rid of trailing newlines, unlike readlines()
        latestts = int(f.read().splitlines()[-1].split(',')[0])
except:
    latestts = 1271779370

def getPushshiftData(a, b):
    url = 'https://api.pushshift.io/reddit/submission/search/?size=500&after='+str(a)+'&before='+str(b)+'&subreddit=Bitcoin&filter=id,author,title,created_utc,domain,num_comments'
    print(url)
    r = requests.get(url)
    r.raise_for_status()
    
    data = json.loads(r.text)
    return data['data']

after, before = latestts, 1618941201 
data = getPushshiftData(after, before)

# from the 'after' date up until before date

while len(data) > 0:
    exp_data = [[c['created_utc'], c['id'], c['author'], c['domain'], c['num_comments'], c['title']] for c in data]
    for c in data:
        print(c)
    with open(exp_file, 'a', newline='\n') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(exp_data)
    print("Saved " + str(len(exp_data)) + " new submissions @ " + exp_file)


    # Calls getPushshiftData() with the created date of the last submission
    after = data[-1]['created_utc']
    print(str(len(data)) + " submissions until " + str(datetime.datetime.fromtimestamp(after)))
    
    while True:
        try:
            data = getPushshiftData(after, before)
            break
        except requests.exceptions.HTTPError as err:
            if err.response.status_code in [525]:
                time.sleep(1)