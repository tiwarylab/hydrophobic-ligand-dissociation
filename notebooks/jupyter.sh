#!/bin/bash


# work around jupyter bug
unset XDG_RUNTIME_DIR

# to help avoid memory issues
export OMP_NUM_THREADS=1

# The ports need to be not otherwise in use.  The 'shuf' command can
# be used to choose ports at random, which almost always works.  Or,
# for convenience, you can choose a fixed port within the given range
# (8000 to 64000), though this may collide more often.
port=$(shuf -i8000-64000 -n1)
# port=12345

hostport=$(shuf -i8000-64000 -n1)
# hostport=12345

node=$(hostname -s)
user=$(whoami)

loginnode=${SLURM_SUBMIT_HOST}

# print tunneling instructions to above output log
cat <<EOF
##########################################################

For MacOS or Linux, use this command to create SSH tunnel:

    ssh -N -f -L localhost:${port}:${node}:${hostport} ${user}@${loginnode}

For Windows/MobaXterm, use this info:

    Forwarded port: ${port}
    Remote server: ${node}
    Remote port: ${hostport}
    SSH server: ${loginnode}
    SSH login: ${user}
    SSH port: 22

Then use this URL to access Jupyter (NOT the one in the log below):

    http://localhost:${port}

##########################################################
EOF

module purge

# uncomment just one: anaconda3 is best for most purposes
# module load anaconda3
# module load python3
# module load anaconda2
# module load python2

# uncomment just one: lab is the new interface, but otherwise compatible with notebook
#jupyter lab --no-browser --port=${hostport} --port-retries=0 --ip='*' --NotebookApp.shutdown_no_activity_timeout=600
jupyter notebook --no-browser --port=${hostport} --port-retries=0 --ip='*' --NotebookApp.shutdown_no_activity_timeout=600
