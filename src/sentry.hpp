#ifndef SENTRY_H__
#define SENTRY_H__

/**
# Fastdup c++ source code
# (c) Dr. Danny Bickson and Dr. Amir Alush.
# File to collect configuration and performance in case of crash using sentry
# For debugging the crash we collect the current config (for example run_mode, feature vector length etc.)
# There is no perfosnal or identifiable information collected.
*/

#include <string>
#include <stdio.h>
#include <sentry.h>
#include <thread>
#include <mutex>
#include <vector>
#include <iomanip>
#include "definitions.h"
#include "performance.h"
#include "logger.hpp"
#ifdef _WIN32
#include <locale>
#include <codecvt>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>

//std::string str = conv.to_bytes(wstr);
//sentry_value_set_by_key(config, "model_path", sentry_value_new_string(str.c_str()));
#endif

#define SENTRY_FASTDUP_VERSION "fastdup@0.211"

using namespace std;

bool is_regular_file(const char * name, bool flag);
bool ends_with(std::string const & value, std::string const & ending);
template <typename T>
int read_filenames(T filename, vector<T>& image_file_paths, bool is_video, bool is_csv, T data_dir,
    size_t thread_id, bool _is_tar);
int increment_run_count(bool flag, int & sentry_count);

void fastdup_sentry_init(TCharString work_dir) {
  
  //run_sentry can be turned of either by turi_param='run_sentry=0' or by defining an env variable SENTRY_OPT_OUT
  if (run_sentry == 0)
    return;

  run_sentry = is_sentry_enabled();
  if (run_sentry == 0)
    return;

  int sentry_report_count = 0;
  increment_run_count(true, sentry_report_count);

#ifndef _WIN32
  string home = getenv("HOME") ? getenv("HOME") : ".";
#else
  wstring home = _wgetenv(L"USERPROFILE") ? _wgetenv(L"USERPROFILE") : L".";
#endif
  home.pop_back();
#ifndef _WIN32
  if (!is_regular_file((home + "/.token").c_str(), false)) {
      token = exec((string("cat ") + home + "/.token").c_str());
  }
  else if (!is_regular_file("/tmp/.token", false)) {
      token = exec("cat /tmp/.token");
  }
#else
  
   wstring token_path = home + L"\\.token";
  if (!is_regular_file(safe_convert(token_path).c_str(), false)) {
      //_tprintf(_T("Failed to find token file\n"));
      vector<wstring> wide_v;
      read_filenames<wstring>(token_path, wide_v, false, true, home, 0, false);
      if (wide_v.size() == 1)
          token = safe_convert(wide_v[0]);
  }
  
#endif
  if (ends_with(token, "\n"))
    token.pop_back();


  sentry_options_t *options = sentry_options_new();
  sentry_options_set_dsn(options, "https://b4723526f0904a27a795406d25f3cc79@o4504135122944000.ingest.sentry.io/4504135125172224");

  // This is also the default-path. For further information and recommendations:
  // https://docs.sentry.io/platforms/native/configuration/options/#database-path
#ifdef _WIN32
  mystring save_path = work_dir + L"\\.__fastdup-cache__";
  //printf("Save path %ls\n", save_path.c_str());
#else
  mystring save_path = work_dir + "/.__fastdup-cache__";

#endif
  sentry_options_set_database_path(options, safe_convert(save_path).c_str());
  sentry_options_set_release(options, SENTRY_FASTDUP_VERSION);
  sentry_options_set_debug(options, 0);
#if defined(__APPLE__) || defined(_WIN32)


  if (getenv("SENTRY_CRASHPAD") != NULL) {
      sentry_options_set_handler_path(options, getenv("SENTRY_CRASHPAD"));
      if (verbose)
        cout << "Using crashpad handler: " << getenv("SENTRY_CRASHPAD") << endl;
      if (verbose && !is_regular_file(getenv("SENTRY_CRASHPAD"), false)) {
          cout << "Missing file crashpad handler" << endl;
      }
  }
  else if (getenv("FASTDUP_LOCAL_DIR") != NULL){
#ifdef _WIN32
      string crashpad = string(getenv("FASTDUP_LOCAL_DIR")) + "\\lib\\crashpad_handler.exe";
#else
      string crashpad = string(getenv("FASTDUP_LOCAL_DIR")) + "/lib/crashpad_handler";

#endif
      if (!is_regular_file(crashpad.c_str(), false)) {
        sentry_options_set_handler_path(options, crashpad.c_str());
        if (verbose)
          cout << "Using crashpad handler from fastdup local dir: " << crashpad << endl;
      }
      else {
          if (verbose)
              cout << "Failed to find crashpad handler from fastdup local dir " << crashpad << endl;
      }
  }

#endif
  sentry_init(options);

    sentry_value_t user = sentry_value_new_object();
    //sentry_value_set_by_key(user, "ip_address", sentry_value_new_string("{{auto}}"));
    sentry_value_set_by_key(user, "id", sentry_value_new_string(token.c_str()));
    sentry_set_user(user);
}

void fastdup_sentry_close(){
  if (run_sentry == 0)
    return;
  sentry_close();
}

void fastdup_sentry_report_config(){
  if (run_sentry == 0)
    return;

    sentry_value_t config = sentry_value_new_object();
    sentry_value_set_by_key(config, "run_mode", sentry_value_new_int32(run_mode));
    sentry_value_set_by_key(config, "d", sentry_value_new_int32(d));
    sentry_value_set_by_key(config, "k", sentry_value_new_int32(nearest_neighbors_k));
    sentry_value_set_by_key(config, "num_clusters", sentry_value_new_int32((int)num_clusters));
    sentry_value_set_by_key(config, "num_em_iter", sentry_value_new_int32((int)num_em_iter));
    sentry_value_set_by_key(config, "num_threads", sentry_value_new_int32(num_threads));
    sentry_value_set_by_key(config, "threshold", sentry_value_new_double(nnthreshold));
    sentry_value_set_by_key(config, "ccthreshold", sentry_value_new_double(ccthreshold[0]));
    sentry_value_set_by_key(config, "lower_threshold", sentry_value_new_double(lower_threshold));
    sentry_value_set_by_key(config, "nn_provider", sentry_value_new_string(nn_provider.c_str()));
    sentry_value_set_by_key(config, "turi_param", sentry_value_new_string(turi_param.c_str()));
    sentry_value_set_by_key(config, "nnf_param", sentry_value_new_string(nnf_params.c_str()));
#ifndef _WIN32
    sentry_value_set_by_key(config, "model_path", sentry_value_new_string(model_path.c_str()));
    sentry_value_set_by_key(config, "input_dir", sentry_value_new_string(data_dir.c_str()));
    sentry_value_set_by_key(config, "work_dir", sentry_value_new_string(work_dir.c_str()));
    sentry_value_set_by_key(config, "install_path", sentry_value_new_string(ld_library_path.c_str()));
#else
    sentry_value_set_by_key(config, "model_path", sentry_value_new_string(conv.to_bytes(model_path).c_str()));
    sentry_value_set_by_key(config, "input_dir", sentry_value_new_string(conv.to_bytes(data_dir).c_str()));
    sentry_value_set_by_key(config, "work_dir", sentry_value_new_string(conv.to_bytes(work_dir).c_str()));
    sentry_value_set_by_key(config, "install_path", sentry_value_new_string(conv.to_bytes(ld_library_path).c_str()));
#endif
    sentry_value_set_by_key(config, "verbose", sentry_value_new_bool(verbose));
   


    sentry_set_context("config", config);
}

void fastdup_sentry_report_file_size(){
  if (run_sentry == 0)
    return;

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
    sentry_value_set_by_key(performance, "runtime_sec", sentry_value_new_int32((int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() -start_time).count()));
    sentry_value_set_by_key(performance, "system_mem_mb", sentry_value_new_int32((int)(getTotalSystemMemory()/1000000)));
    sentry_value_set_by_key(performance, "cores", sentry_value_new_int32((int)std::thread::hardware_concurrency()));
    sentry_value_set_by_key(performance, "disk_space", sentry_value_new_string(get_disk_space().c_str()));
    sentry_value_set_by_key(performance, "has_cuda", sentry_value_new_bool(perf_report.is_cuda));
    sentry_value_set_by_key(performance, "is_jupyter", sentry_value_new_bool(perf_report.is_jupyter));
    sentry_value_set_by_key(performance, "is_ec2", sentry_value_new_bool(perf_report.is_ec2));
    sentry_value_set_by_key(performance, "is_arm", sentry_value_new_bool(perf_report.is_arm));
    sentry_value_set_by_key(performance, "is_docker", sentry_value_new_bool(perf_report.is_docker));
    sentry_value_set_by_key(performance, "is_google", sentry_value_new_bool(perf_report.is_google));
    sentry_value_set_by_key(performance, "is_databricks", sentry_value_new_bool(perf_report.is_databricks));
    sentry_value_set_by_key(performance, "is_sagemaker", sentry_value_new_bool(perf_report.is_sagemaker));
    sentry_value_set_by_key(performance, "is_wsl", sentry_value_new_bool(perf_report.is_wsl));
    
    sentry_set_context("performance", performance);

    sentry_set_tag("is_ec2", perf_report.is_ec2? "yes": "no");
    sentry_set_tag("is_google", perf_report.is_google? "yes": "no");
    sentry_set_tag("is_jupyter", perf_report.is_jupyter? "yes": "no");
    sentry_set_tag("is_databricks", perf_report.is_databricks? "yes": "no");
    sentry_set_tag("is_docker", perf_report.is_docker? "yes":"no");
    sentry_set_tag("is_wsl", perf_report.is_wsl? "yes":"no");
    sentry_set_tag("is_arm", perf_report.is_arm? "yes":"no");
    sentry_set_tag("is_cuda", perf_report.is_cuda? "yes":"no");
    sentry_set_tag("server_name", perf_report.hostname.c_str());

    sentry_set_tag("unit_test", perf_report.is_unittest? "True": "False");
    sentry_set_tag("is_sagemaker", perf_report.is_sagemaker? "yes": "no");
    sentry_set_tag("over_million", num_images >= 1000000 ? "yes": "no");
    sentry_set_tag("is_licensed", license != "" ? "yes": "no");
    sentry_set_tag("token", token.c_str());
    float secs = (float)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() -start_time).count();
    sentry_set_tag( "runtime", std::to_string(secs).c_str());
    sentry_set_tag( "device", perf_report.device.c_str());


    sentry_value_t admin = sentry_value_new_object();
    sentry_value_set_by_key(admin, "license", sentry_value_new_string(license.c_str()));
    sentry_set_context("admin", admin);

}


void fastdup_sentry_report(string logger, string message, sentry_level_t level=SENTRY_LEVEL_ERROR){
  if (run_sentry == 0)
    return;

 /* ... */
  try{
    fastdup_sentry_report_file_size();
    sentry_capture_event(sentry_value_new_message_event(
  /*   level */ level,
  /*  logger */ logger.c_str(),
  /* message */ message.c_str()
   ));
  }
  catch(...){

  };

}

#include <stdarg.h>

void fastdup_sentry_report_error_msg(const char * error_type, const char * format, ...){

   
    va_list args;

    char msg[2056];
    va_start(args, format);
#ifndef _WIN32
    vsprintf(msg, format, args);
#else
    vsprintf_s(msg, format, args);
#endif

    va_end(args);

	if (images_too_small_count > 10 && string(error_type) == "Image Limits"){
		//slow does error messages
	}
    else {
		FASTDUP_ERROR(msg);
		//fprintf(stderr, "%s", msg);
	}
   
    FILE * pfile = NULL;
#ifndef _WIN32
    string filename = work_dir + get_sep_str() + FILENAME_ERROR_MSG;
    if (is_regular_file(filename.c_str(), false))
        pfile = fopen(filename.c_str(), "w");
    else
        pfile = fopen(filename.c_str(), "a");
#else
    int err;
    TCharString filename = work_dir + L"\\" + conv.from_bytes(FILENAME_ERROR_MSG);
    std::string narrow_filename = conv.to_bytes(filename);
    if (is_regular_file(narrow_filename.c_str(), false))
        err = fopen_s(&pfile, narrow_filename.c_str(), "w");
    else
        err = fopen_s(&pfile, narrow_filename.c_str(), "a");

#endif
    if (pfile == NULL){
#ifndef _WIN32
        fprintf(stderr, "Failed to open log file for writing %s\n", filename.c_str());
#else
        fprintf(stderr, "Failed to open log file for writing %ws\n", filename.c_str());
#endif
    }
    else {
        fprintf(pfile, "%s", msg);
        fclose(pfile);
    }

    if (run_sentry == 0)
        return;

	//slow down reporting when there are two many messages on a given day
	int sentry_count_so_far;
	int rcount = increment_run_count(false, sentry_count_so_far);
	if (rcount == -1)
		return;

    if (!log_slow_down && sentry_count_so_far < MAX_ALLOWED_SENTRY_REPORTS_PER_DAY)
        fastdup_sentry_report(error_type, msg);
}
#endif //SENTRY_H__

