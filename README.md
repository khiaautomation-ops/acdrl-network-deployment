Deep Reinforcement Learning Algorithm for Adaptive & Energy-Efficient Deployment of Robotic Airborne Base Stations (RABS)

This repository provides the official technical implementation of the architecture proposed in our IEEE paper on Deep Reinforcement Learning (DRL) for autonomous network infrastructure. This work was developed by our Co-founder at KHIA AI under the supervision of Professors at École de technologie supérieure (ÉTS) and supported by FRQNT (Quebec).

The project implements an adaptive, energy-efficient framework for the deployment of Robotic Airborne Base Stations (RABS). By leveraging DRL, the system dynamically optimizes the placement of airborne nodes to maximize coverage while minimizing energy consumption in fluctuating traffic environments.

Key Technical Features:
1. Dynamic Traffic Profiling: Automated generation of spatio-temporal traffic demand.
2. DRL-Driven Optimization: Implementation of Multi Agents ACDRL for real-time RABS positioning.
3. Energy-Aware Deployment: Specialized reward functions focusing on maximizing battery life and operational efficiency.

Getting Started
1. Environment Setup
We recommend using a virtual environment to manage dependencies:

2. Data Preparation
The system relies on traffic demand profiles and initial placement configurations.
Run the following modules to prepare the environment:
- Traffic Profiles: Generate demand data using the traffic code files, "GenerateTraffic.py"
- RABS Placement: Initialize the spatial coordinates for the airborne stations, "RABS_Placement.py"

3. Execution & Evaluation
Run the different models separately to compare performance metrics (e.g., Energy Efficiency).

📖 Technical Background & Citation
This software is grounded in rigorous, peer-reviewed methodology published in IEEE Transactions on Network and Service Management.

Full Citation:
E. Theingi, L. Sboui and D. Naboulsi, "Adaptive and Energy-Efficient Deployment of Robotic Airborne Base Stations: A Deep Reinforcement Learning Approach," in IEEE Transactions on Network and Service Management, vol. 23, pp. 3707-3721, 2026, doi: 10.1109/TNSM.2026.3678488.

Code snippet
@article{theingi2026rabs,
  title={Adaptive and Energy-Efficient Deployment of Robotic Airborne Base Stations: A Deep Reinforcement Learning Approach},
  author={Theingi, Ei and Sboui, L. and Naboulsi, D.},
  journal={IEEE Transactions on Network and Service Management},
  volume={23},
  pages={3707--3721},
  year={2026},
  publisher={IEEE}
}
