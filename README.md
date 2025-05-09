# DQN Snake

### Overview
This project is a multi-agent and dynamic environment Snake game using Reinforcement Learning and Neural Network.

### Design
In this game, the snake should avoid its head to collide with its own body, the walls, obstacles and eat candy as much as it can grow longer to maximize the scores it achieved. I plan to have 3 levels of this game, with the change of the size of map, the number of obstacles, and the number of candy. 

Environment Features will consist of :
● Dynamic obstacles and walls
● Randomly appearing/disappearing food with different values
● Multiple AI agents competing in same space
● Map size adjusts with number of players

### Reinforcement Learning
**State:**
- Check the state around the snake head: if there are wall, snake body, obstacles, candy
- Other players' positions and movements
- Current snake length & The distance and direction to the nearest candy it detected
- Environment change indicators

**Action:**
- Go up, right, down, left

**Reward:**
- If get candy: +10
- If get closer to candy: + 1
- If get away from candy: -1
- If meet collision: - 100

### Neural network
**Input Layer:**
- Environment state
- Other players' positions
- Food locations
- Current movement state

**Hidden Layers:**
- Multiple layers with ReLU activation
- Dropout for overfitting prevention

**Output Layer:**
- Movement direction probabilities

### Reference
The dynamic environment setting is inspired by *H. F. Gärdström’s work Adaptive Agents in 1v1 Snake Game with Dynamic Environment*.[1] This paper investigates the adaptability of agent trained agents in the dynamic snake game environments, though I simplified their approach to focus on obstacle dynamics rather than full environmental adaptation.

### Author
Nova | Boston | 2024 