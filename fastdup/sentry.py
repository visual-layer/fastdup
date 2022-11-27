#Sentry collects crash reports and performance numbers
#It is possible to turn off data collection using an environment variable named "SENTRY_OPT_OUT"
import sentry_sdk
from sentry_sdk import capture_exception
import time
import os
import traceback
import uuid
import hashlib
from fastdup.definitions import VERSION__
from datetime import datetime


#get a random token based on the machine uuid
token = hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
unit_test = False

def init_sentry():
    global unit_test
    if 'SENTRY_OPT_OUT' not in os.environ:
        sentry_sdk.init(
            dsn="https://b526f209751f4bcea856a1d90e7cf891@o4504135122944000.ingest.sentry.io/4504168616427520",

            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=1.0,
            release=VERSION__
        )
        unit_test = 'UNIT_TEST' in os.environ
        try:
            with open(os.path.join(os.environ.get('HOME', '/tmp'),".token"), "w") as f:
                f.write(token)
        except:
            pass

def fastdup_capture_exception(section, e, warn_only=False):
    if not warn_only:
        traceback.print_exc()
    if 'SENTRY_OPT_OUT' not in os.environ:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("section", section)
            scope.set_tag("unit_test", unit_test)
            scope.set_tag("token", token)
            capture_exception(section, e)


def fastdup_performance_capture(section, start_time):
    if 'SENTRY_OPT_OUT' not in os.environ:
        try:
            # avoid reporting unit tests back to sentry
            if token == '41840345eec72833b7b9928a56260d557ba2a1e06f86d61d5dfe755fa05ade85':
                import random
                if random.random() < 0.995:
                    return
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("section", section)
                scope.set_tag("unit_test", unit_test)
                scope.set_tag("token", token)
                scope.set_extra("runtime-sec", time.time()-start_time)
                sentry_sdk.capture_message("Performance")
        finally:
            sentry_sdk.flush()

