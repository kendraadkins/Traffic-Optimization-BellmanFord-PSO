# Traffic Optimization Using Bellman-Ford and Particle Swarm Optimization

## Overview

This project explores traffic optimization around Marshall University using graph-based algorithms. The campus road network is modeled as a weighted graph, and two approaches are compared:

- **Bellman-Ford Algorithm** (deterministic shortest-path)
- **Particle Swarm Optimization (PSO)** (adaptive optimization)

The goal is to evaluate how different algorithms perform under various real-world traffic conditions.

---

## Motivation

Traffic congestion around Marshall University is heavily influenced by:

- Class change times  
- Pedestrian traffic  
- Special events (e.g., football games)  
- Road closures and construction  
- Emergency or police activity  

This project simulates these conditions and analyzes how routing strategies perform in each case.

---

## Algorithms Used

### Bellman-Ford
- Computes shortest paths in weighted graphs  
- Works well for stable, predictable traffic conditions  
- Guarantees optimal routes when weights are known  

### Particle Swarm Optimization (PSO)
- Population-based optimization inspired by swarm behavior  
- Explores multiple routing possibilities  
- More adaptable to dynamic or changing conditions  

---

## Simulated Scenarios

This project includes multiple traffic scenarios:

- **Normal Traffic**
- **Football Game Day**
- **Pedestrian Heavy Traffic**
- **Construction on 5th Avenue**
- **Heavy Police Activity**

Each scenario modifies edge weights to reflect real-world conditions such as delays, congestion, and restricted routes.

---

## Results

Key findings from the simulations:

- Bellman-Ford consistently produced shorter travel times  
- Largest improvement (~16.88%) occurred in high-stress scenarios  
- Bellman-Ford had slightly better runtime performance  
- Memory usage was similar across both algorithms  
- PSO showed greater flexibility but did not always reach optimal solutions  

---

## Technologies Used

- Python  
- NetworkX  
- Matplotlib  

---

## How to Run

Example:

```bash
python normalPath/main.py
