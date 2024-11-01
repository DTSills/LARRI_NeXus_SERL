export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env PrinterEnv \
    --exp_name=serl_printer_sim_test \
    --seed 0 \
    --random_steps 1000 \
    --debug
