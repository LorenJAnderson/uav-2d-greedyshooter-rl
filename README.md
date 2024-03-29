# uav-2d-greedyshooter-rl

Reinforcement learning environment and algorithm for the 2-dimensional greedy shooter problem. 
Companion code for the paper "On Solving the 2-Dimensional Greedy Shooter Problem for UAVs" at arXiv:1911.01419.


Call `python dqn.py` or optionally `python dqn.py --cuda` to run. 
Tensorboard log file will be created in a newly created `runs` folder.
Training and testing of models occurs simultaneously.
Folder `Test` will be created in current directory to house the best pytorch 
    models during testing.
Program will finish once testing reaches a 100% win rate.

Training and testing reward dynamics are plotted by modifying the script `train_test_graph.py` and calling
    `python train_test_graph.py`. This creates a `graph` folder with the dynamics figure in .pdf format.
    
Trajectories are plotted by modifying the script `trajectory_plotter.py` and calling
    `python trajectories_plotter.py`. This creates a `trajs` folder with 80 trajectory figures in .pdf format.
    
The reinforcement learning architecture is adopted from the following: 
    Lapan, M. (2018). Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks,
    value iteration, policy gradients, TRPO, AlphaGo Zero and more. Packt Publishing Ltd.