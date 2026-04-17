# Deep Reinforcement Learning for Adaptive & Energy-Efficient RABS Deployment

[![IEEE Xplore](https://img.shields.io/badge/Paper-IEEE%20TNSM-blue)](https://doi.org/10.1109/TNSM.2026.3678488)

This repository contains the official technical implementation of the architecture proposed in our IEEE paper regarding **Deep Reinforcement Learning (DRL)** for autonomous network infrastructure.

Developed by the Co-founder of **KHIA AI** under the supervision of professors at **École de technologie supérieure (ÉTS)** and supported by **FRQNT (Quebec)**.

> [!WARNING]
> **Research Code Disclaimer:** This is the original research implementation code used for our IEEE paper. It is provided 'as-is' for educational purposes and has not been refactored for production-grade software environments.

---

## Overview

The project implements an adaptive, energy-efficient framework for the deployment of **Robotic Airborne Base Stations (RABS)**. By leveraging DRL, the system dynamically optimizes the placement of airborne nodes to:
* **Maximize Coverage:** Adapting to fluctuating spatio-temporal traffic.
* **Optimize Energy:** Specialized reward functions focused on battery life and operational efficiency.

### Key Technical Features
1.  **Dynamic Traffic Profiling:** Automated generation of spatio-temporal traffic demand.
2.  **DRL-Driven Optimization:** Implementation of **Multi-Agent ACDRL** for real-time RABS positioning.
3.  **Energy-Aware Deployment:** Intelligent power management within the DRL agent's logic.

---

## Getting Started
### 1. Environment Setup
We recommend using a virtual environment to manage dependencies:

### **2. Data Preparation**
The system relies on traffic demand profiles and initial placement configurations. Run the following modules to prepare the environment:
Traffic Profiles: Generate demand data.
Bash
python GenerateTraffic.py
RABS Placement: Initialize spatial coordinates for the airborne stations.
Bash
python RABS_Placement.py

### **3. Execution & Evaluation**
Run the individual DRL model modules to compare performance metrics such as Energy Efficiency and Coverage Ratio across different scenarios.

📖 Citation & Technical Background
This software is grounded in the peer-reviewed methodology published in IEEE Transactions on Network and Service Management.

**Full Citation:**
E. Theingi, L. Sboui and D. Naboulsi, "Adaptive and Energy-Efficient Deployment of Robotic Airborne Base Stations: A Deep Reinforcement Learning Approach," in IEEE Transactions on Network and Service Management, vol. 23, pp. 3707-3721, 2026, doi: 10.1109/TNSM.2026.3678488.

BibTeX
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
