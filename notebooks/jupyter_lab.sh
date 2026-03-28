export SCR=/w/20252/wjcai/uq/jupyter_folder
mkdir -p "$SCR"/{jupyter_data,jupyter_runtime,jupyter_config,tmp,project}

export JUPYTER_DATA_DIR="$SCR/jupyter_data"
export JUPYTER_RUNTIME_DIR="$SCR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$SCR/jupyter_config"
export TMPDIR="$SCR/tmp"

# cd "$SCR/project"
jupyter lab --no-browser --ip=0.0.0.0 --port=8888