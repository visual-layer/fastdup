#ifndef SENTRY_H__
#define SENTRY_H__

# Fastdup c++ source code
# (c) Dr. Danny Bickson and Dr. Amir Alush.
# File to collect configuration and performance in case of crash using sentry
# For debugging the crash we collect the current config (for example run_mode, feature vector length etc.)
# There is no perfosnal or identifiable information collected.

#include <string>
#ifndef APPLE
#include <sentry.h>
#endif //APPLE
#include "definitions.h"
#include "performance.h"

using namespace std;
void fastdup_sentry_init(void) {

  //run_sentry can be turned of eitehr by turi_param='run_sentry=0' or by defining an env variable SENTRY_OPT_OUT
  if (run_sentry == 0)
    return;

  run_sentry = is_sentry_enabled();
#ifndef APPLE
  sentry_options_t *options = sentry_options_new();
  sentry_options_set_dsn(options, "https://b4723526f0904a27a795406d25f3cc79@o4504135122944000.ingest.sentry.io/4504135125172224");

  // This is also the default-path. For further information and recommendations:
  // https://docs.sentry.io/platforms/native/configuration/options/#database-path
  sentry_options_set_database_path(options, ".sentry-native");
  sentry_options_set_release(options, "fastdup@0.155");
  sentry_options_set_debug(options, 1);
  sentry_init(options);
#endif //APPLE
}

void fastdup_sentry_close(){
  if (run_sentry == 0)
    return;
#ifndef APPLE
  sentry_close();
#endif
}

void fastdup_sentry_report(string logger, string message, sentry_level_t level=SENTRY_LEVEL_ERROR){
  if (run_sentry == 0)
    return;

#ifndef APPLE
 /* ... */
  try{
    sentry_capture_event(sentry_value_new_message_event(
  /*   level */ level,
  /*  logger */ logger.c_str(),
  /* message */ message.c_str()
   ));
  }
  catch(...){

  };
#endif

}

void fastdup_sentry_report_config(){
  if (run_sentry == 0)
    return;

#ifndef APPLE
    sentry_value_t config = sentry_value_new_object();
    sentry_value_set_by_key(config, "run_mode", sentry_value_new_int32(run_mode));
    sentry_value_set_by_key(config, "d", sentry_value_new_int32(d));
    sentry_value_set_by_key(config, "k", sentry_value_new_int32(nearest_neighbors_k));
    sentry_value_set_by_key(config, "num_clusters", sentry_value_new_int32(num_clusters));
    sentry_value_set_by_key(config, "num_em_iter", sentry_value_new_int32(num_em_iter));
    sentry_value_set_by_key(config, "num_threads", sentry_value_new_int32(num_threads));
    sentry_value_set_by_key(config, "threshold", sentry_value_new_double(nnthreshold));
    sentry_value_set_by_key(config, "ccthreshold", sentry_value_new_double(ccthreshold));
    sentry_value_set_by_key(config, "lower_threshold", sentry_value_new_double(lower_threshold));
    sentry_value_set_by_key(config, "nn_provider", sentry_value_new_string(nn_provider.c_str()));
    sentry_value_set_by_key(config, "turi_param", sentry_value_new_string(turi_param.c_str()));
    sentry_value_set_by_key(config, "nnf_param", sentry_value_new_string(nnf_params.c_str()));
    sentry_value_set_by_key(config, "verbose", sentry_value_new_bool(verbose));
    sentry_set_context("config", config);
#endif
}

void fastdup_sentry_report_file_size(){
  if (run_sentry == 0)
    return;

#ifndef APPLE
    sentry_value_t data = sentry_value_new_object();
    sentry_value_set_by_key(data, "num_images", sentry_value_new_int32((int)num_images));
    sentry_value_set_by_key(data, "num_images_test", sentry_value_new_int32((int)num_images_test));
    sentry_value_set_by_key(data, "is_minio", sentry_value_new_bool(global_minio_config.is_minio));
    sentry_value_set_by_key(data, "is_s3", sentry_value_new_bool(global_minio_config.is_s3));
    sentry_value_set_by_key(data, "found_video", sentry_value_new_bool(found_video));
    sentry_value_set_by_key(data, "found_tar", sentry_value_new_bool(found_tar));
    sentry_value_set_by_key(data, "found_zip", sentry_value_new_bool(found_zip));

    sentry_set_context("data", data);

    sentry_value_t performance = sentry_value_new_object();
    sentry_value_set_by_key(performance, "runtime_sec", sentry_value_new_int32(duration_cast<seconds>(system_clock::now() -start_time).count()));
    sentry_value_set_by_key(performance, "system_mem_mb", sentry_value_new_int32((int)(getTotalSystemMemory()/1000000)));
    sentry_value_set_by_key(performance, "cores", sentry_value_new_int32((int)std::thread::hardware_concurrency()));
    sentry_value_set_by_key(performance, "disk_space", sentry_value_new_string(get_disk_space().c_str()));
    sentry_set_context("performance", performance);


#endif //APPLE
}

#include <stdarg.h>

void fastdup_sentry_report_error_msg(const char * error_type, const char * format, ...){

    int result;
    va_list args;

    char msg[2056];
    va_start(args, format);
    vsprintf(msg, format, args);
    va_end(args);

    fprintf(stderr, "%s", msg);
    if (run_sentry == 0)
        return;

#ifndef APPLE

    fastdup_sentry_report(error_type, msg);
#endif //APPLE
}
#endif //SENTRY_H__

