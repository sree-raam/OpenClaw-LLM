# OpenClaw-LLM

A latency-aware ROS 2 framework integrating Large Language Models (LLMs) for intelligent task planning, autonomous navigation, safety supervision, and benchmark evaluation on edge robotic platforms.

---

## Overview

OpenClaw-LLM is a Final Year Project developed at Universiti Teknologi Malaysia (UTM), MJIIT. The framework investigates the feasibility of deploying lightweight LLMs for robotic task planning under edge-computing constraints using ROS 2 and NVIDIA Jetson Orin Nano.

## Features

- Natural Language Task Commands
- LLM-Based Mission Planning
- ROS 2 Autonomous Navigation
- Safety Monitoring and Recovery
- Latency-Aware Performance Evaluation
- Multi-LLM Benchmarking
- Edge AI Deployment Support

## Supported Models

| Model | Parameters |
|-------|------------|
| Gemma2 | 2B |
| Qwen2.5 | 3B |
| DeepSeek-R1 | 1.5B |

## Technology Stack

- ROS 2 Humble
- Gazebo Classic
- TurtleBot3 Burger
- Ollama
- Python 3
- NVIDIA Jetson Orin Nano 8GB
- Ubuntu 22.04

## Repository Structure

```text
OpenClaw-LLM
│
├── openclaw_controller_v25_final_recovery_supervisor.py
├── README.md
├── LICENSE
├── OVERALL FLOW.png
├── HARDWARE SET UP.png
├── RESULT BENCHMARK 40.docx
└── FYP2 Technical Paper.pdf
```

## Evaluation Metrics

The framework evaluates:

- LLM inference latency
- Navigation completion time
- Success rate
- CPU utilization
- Memory consumption
- Path efficiency
- Recovery effectiveness

## Future Work

- Real robot deployment
- Isaac Sim integration
- Vision-Language Models
- Dynamic obstacle avoidance
- Multi-agent coordination

## Author

**Sree Raam**  
Bachelor of Electronic Systems Engineering (Hons.)  
Malaysia-Japan International Institute of Technology (MJIIT)  
Universiti Teknologi Malaysia

Supervisor:  
Assoc. Prof. Ir. Dr. Zool Hilmi bin Ismail

## License

This project is released for academic and research purposes.
