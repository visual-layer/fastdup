import datetime
import pandas as pd
import os
from urllib import request
import sys
import hashlib
import uuid

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
            return True

    except Exception as ex:
        return True


# send a get query for the latest version
# this query has no fields and do not share information about the client besides the system version so we 
# could match the right fastdup version to the current platform see comment below)
def check_for_update(version):
    need_update = need_to_check_for_update()
    if not need_update:
        return

    #create a unique random request string
    uniq = hashlib.sha1(hex(uuid.getnode()).encode('utf-8')).hexdigest()

    # check for the latest fastdup version based on the system version
    # (fastdup package c++ code thus platform like libc version is critical for getting the right fastdup version.
    # Currently we support PEP 2_31 and 2_27 platforms via pypi and 2_17 via our release page.
    # Read more about paltforms and PEP standard here: https://github.com/pypa/auditwheel
    ver = sys.version
    ver = ''.join(ch for ch in ver if (ch.isalnum() or ch  == '.' or ch == ',' or ch == ':'))
    url = f"https://databasevisual.com/_functions/latestfastdupversion?sys={ver}&uuid={uniq}&curversion={version}"

    try:
        req = request.Request(url)
        ret = request.urlopen(req, timeout=3)
        # answer is of the form {'version':'0.108'}
        if 'version' in ret.headers:
            if version != ret.headers['version']:
                if float(version) < float(ret.headers['version']):
                	print(f"Warning: detected a newer version of fastdup {ret.headers['version']}, your version is {version}. Please update using: python3.8 -m pip install -U fastdup.")
        
            cur_date = str(datetime.datetime.now().date())
            pd.DataFrame({'last_run':[cur_date]}).to_csv('/tmp/lastrun')

    except Exception as ex:
        return

