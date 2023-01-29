#Sentry collects crash reports and performance numbers
#It is possible to turn off data collection using an environment variable named "SENTRY_OPT_OUT"
import sentry_sdk
from sentry_sdk import capture_exception

import time
import os
import traceback
import platform
import uuid
import hashlib
from fastdup.definitions import VERSION__


#get a random token based on the machine uuid
token = hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
unit_test = None


def find_certifi_path():
    try:
        import certifi
        return os.path.join(os.path.dirname(certifi.__file__), 'cacert.pem')
    except Exception as ex:
        print('Failed to find certifi', ex)
    return None


def traces_sampler(sampling_context):
    # Examine provided context data (including parent decision, if any)
    # along with anything in the global namespace to compute the sample rate
    # or sampling decision for this transaction

    print(sampling_context)
    return 1

def init_sentry():
    global unit_test

    if 'SENTRY_OPT_OUT' not in os.environ:

        if platform.system() == 'Darwin':
            # fix CA certficate issue on latest MAC models
            path = find_certifi_path()
            if path is not None:
                if 'SSL_CERT_FILE' not in os.environ:
                    os.environ["SSL_CERT_FILE"] = path
                if 'REQUESTS_CA_BUNDLE' not in os.environ:
                    os.environ["REQUESTS_CA_BUNDLE"] = path

        sentry_sdk.init(
            dsn="https://b526f209751f4bcea856a1d90e7cf891@o4504135122944000.ingest.sentry.io/4504168616427520",
            debug='SENTRY_DEBUG' in os.environ,
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=1,
            release=VERSION__,
            default_integrations=False
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
            capture_exception(e, scope=scope)


def fastdup_performance_capture(section, start_time):
    if 'SENTRY_OPT_OUT' not in os.environ:
        try:
            # avoid reporting unit tests back to sentry
            if token == '41840345eec72833b7b9928a56260d557ba2a1e06f86d61d5dfe755fa05ade85':
                import random
                if random.random() < 0.995:
                    return
            sentry_sdk.set_tag("runtime", str(time.time()-start_time))

            with sentry_sdk.push_scope() as scope:
                scope.set_tag("section", section)
                scope.set_tag("unit_test", unit_test)
                scope.set_tag("token", token)
                scope.set_tag("runtime-sec", time.time()-start_time)
                sentry_sdk.capture_message("Performance", scope=scope)
        finally:
            sentry_sdk.flush(timeout=5)


def fastdup_capture_log_debug_state(config):
    if 'SENTRY_OPT_OUT' not in os.environ:
        breadcrumb = {'type':'debug', 'category':'setup', 'message':'snapshot', 'level':'info', 'timestamp':time.time() }
        breadcrumb['data'] = config
        with sentry_sdk.configure_scope() as scope:
            scope.clear_breadcrumbs()
        sentry_sdk.add_breadcrumb(breadcrumb)
