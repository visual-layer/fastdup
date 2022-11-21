#Sentry collects crash reports and performance numbers
#It is possible to turn off data collection using an environment variable named "SENTRY_OPT_OUT"
import sentry_sdk
from sentry_sdk import capture_exception
import time
import os
import traceback
import uuid
import hashlib

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
            traces_sample_rate=1.0
        )
        unit_test = 'UNIT_TEST' in os.environ
        with open(os.path.join(os.environ.get('HOME', '/tmp'),".token"), "w") as f:
            f.write(token)


def fastdup_capture_exception(section, e, warn_only=False):
    if not warn_only:
        traceback.print_exc()
    if 'SENTRY_OPT_OUT' not in os.environ:
        fastdup_performance_capture(section, time.time())
        capture_exception(e)


def fastdup_performance_capture(section, start_time):
    if 'SENTRY_OPT_OUT' not in os.environ:
        try:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("section", section)
                scope.set_tag("unit_test", unit_test)
                scope.set_tag("token", token)
                scope.set_extra("runtime", time.time()-start_time)
                sentry_sdk.capture_message("Performance")
        finally:
            sentry_sdk.flush()

