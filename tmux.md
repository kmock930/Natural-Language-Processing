# Guide to use tmux on a remote machine
This project involves long processes and as a result, **tmux** becomes a handy tool essentially for idling long processes like python files on a remote virtual machine. Here is the guide of using **tmux**. 

-----

## Steps
1. Run this command to launch a new session named `trainjob` (for example) on tmux: `tmux new -s trainjob`.
2. As a good practice, create a virtual environment and install necessary dependencies with this command: `python3 -m venv [environment_name]`.
3. Run this command to activate the virtual environment: (Linux) `source venv/bin/activate`; (Windows) `./venv/Scripts/activate`.
4. Run your python script on tmux with this command: `python3 train.py`.
* Note: If you intend to keep track of logs on the terminal, please run this command `python3 train.py > train.log 2>&1`, and you could inspect the log file anytime with this command `tail -f train.log`.

## Other useful **tmux** commands
* To list all the tmux sessions: `tmux ls`.

2. To attach (go back/load an existing session): `tmux attach -t (session name/no.)`.

3. To kill a tmux session: `tmux kill-session`.

4) To kill a tmux server: `tmux kill-server`.