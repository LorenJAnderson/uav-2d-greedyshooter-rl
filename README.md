# uav-2d-greedyshooter-rl

Reinforcement learning environment and algorithm for the 2-dimensional greedy shooter problem. 
Companion code for the paper 


Call `python dqn.py` or optionally `python dqn.py --cuda` to run. 
Tensorboard log file will be created in a newly created `runs` folder.
Training and testing of models occurs simultaneously.
Folder `Test` will be created in current directory to house the best pytorch 
    models during testing.
Program will finish once testing reaches a 100% win rate.
Trajectories can be plotted by modifying the script `trajectory_plotter.py` and calling
    `python trajectory_plotter.py`. 
This creates a `pics` folder with 80 trajectory figures in .jpg format.