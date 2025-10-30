![Isaac Lab](docs/source/_static/isaaclab.jpg)
## Getting Started

### Getting Started with Open-Source Isaac Sim

Isaac Sim is now open source and available on GitHub!

For detailed Isaac Sim installation instructions, please refer to
[Isaac Sim README](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#quick-start).

1. Clone Isaac Sim

    ```
    git clone https://github.com/isaac-sim/IsaacSim.git
    ```

2. Build Isaac Sim

    ```
    cd IsaacSim
    ./build.sh
    ```

    On Windows, please use `build.bat` instead.

3. Clone Isaac Lab

    ```
    cd ..
    git clone https://github.com/manggoF/IsaacLab-RL.git
    cd IsaacLab-RL
    ```

4. Set up symlink in Isaac Lab

    Linux:

    ```
    ln -s ${ISAACSIM_PATH} _isaac_sim
    ```

5. Install Isaac Lab

    Linux:

    ```
    ./isaaclab.sh -i
    ```

    Windows:

    ```
    isaaclab.bat -i
    ```

6. [Optional] Set up a virtual python environment (e.g. for Conda)

    Linux:

    ```
    # Option 1: Default environment name 'env_isaaclab'
    ./isaaclab.sh --conda  # or "./isaaclab.sh -c"
    # Option 2: Custom name
    ./isaaclab.sh --conda my_env  # or "./isaaclab.sh -c my_env"
    ```
    
    vscode配置 ：
    
    ```
    Run VSCode Tasks, by pressing Ctrl+Shift+P, selecting Tasks: Run Task and running the setup_python_env in the drop down menu.
    ```


7. Train!

    Linux:

    ```
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Ant-v0 --headless
    ```

    Windows:

    ```
    isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task Isaac-Ant-v0 --headless
    ```
