import datetime
import pandas as pd
import os
from urllib import request

# limit the number of times we check for updates to at most once per day
def need_to_check_for_update():
    try:
        cur_date = str(datetime.datetime.now().date())
        if os.path.exists('/tmp/lastrun'):
            lastrun = pd.read_csv('/tmp/lastrun')['last_run'].values[0]
            if lastrun != cur_date:
                return True;
            else:
                return False;
        else:
            pd.DataFrame({'last_run':[cur_date]}).to_csv('/tmp/lastrun')
            return True

    except Exception as ex:
        return False


# send a get query for the latest version
# this query has no fields and do not share information about the client (os, python version, current version etc)
def check_for_update(version):
    need_update = need_to_check_for_update()
    if not need_update:
        return

    url = 'https://databasevisual.com/_functions/latestfastdupversion'

    try:
        req = request.Request(url)
        ret = request.urlopen(req, timeout=1)
        # answer is of the form {'version':'0.108'}
        if 'version' in ret.headers:
            if version != ret.headers['version']:
                print(f"Warning: detected a newer version of fastdup {ret.headers['version']}, your version is {version}. Please update using: python3.8 -m pip install -U fastdup.")

    except Exception as ex:
        return

