import random

class MadHatterCasino:
    def __init__(self):
        self.room = 0
        self.return_values = [] # return_values[action][state][noise]
        action0 = []
        action1 = []
        action2 = [] # action0[state][noise]
        action0.append([[0,3.1],
                        [0,2.9],
                        [1,3.2],
                        [1,2.8],
                        [2,3.0],
                        [1,3.0],
                        [1,3.1],
                        [1,2.9],
                        [3,3.2],
                        [3,3.1]])
        action0.append([[0,4.0],
                        [0,4.0],
                        [0,4.0],
                        [0,4.1],
                        [0,3.9],
                        [0,3.8],
                        [0,4.2],
                        [0,3.9],
                        [1,4.0],
                        [2,3.8]])
        action0.append([[0,3.0],
                        [0,3.0],
                        [0,3.1],
                        [0,2.8],
                        [0,2.7],
                        [0,3.3],
                        [1,3.4],
                        [2,3.0],
                        [3,3.0],
                        [3,3.1]])
        action0.append([[2,3.8],
                        [2,4.1],
                        [2,4.0],
                        [2,4.0],
                        [2,3.7],
                        [2,4.2],
                        [2,4.0],
                        [2,3.7],
                        [3,4.4],
                        [3,4.0]])
        action1.append([[0,1.1],
                        [1,0.9],
                        [1,1.2],
                        [1,0.8],
                        [1,1.0],
                        [1,1.0],
                        [1,1.1],
                        [1,0.9],
                        [1,1.2],
                        [3,1.1]])
        action1.append([[0,2.0],
                        [1,2.0],
                        [2,2.0],
                        [2,1.1],
                        [2,1.9],
                        [2,1.8],
                        [2,2.2],
                        [2,1.9],
                        [2,2.0],
                        [2,1.8]])
        action1.append([[1,1.0],
                        [2,1.0],
                        [3,1.1],
                        [3,0.8],
                        [3,0.7],
                        [3,1.3],
                        [3,1.4],
                        [3,1.0],
                        [3,1.0],
                        [3,1.1]])
        action1.append([[0,1.8],
                        [0,2.1],
                        [0,2.0],
                        [0,2.0],
                        [0,1.7],
                        [0,2.2],
                        [0,2.0],
                        [0,1.7],
                        [0,2.4],
                        [3,2.0]])
        action2.append([[1,3.1],
                        [2,2.9],
                        [2,3.2],
                        [2,2.8],
                        [3,3.0],
                        [3,3.0],
                        [3,3.1],
                        [3,2.9],
                        [3,3.2],
                        [3,3.1]])
        action2.append([[0,3.0],
                        [1,3.0],
                        [1,3.0],
                        [1,3.1],
                        [1,2.9],
                        [1,2.8],
                        [1,3.2],
                        [1,2.9],
                        [1,3.0],
                        [3,2.8]])
        action2.append([[0,3.0],
                        [2,3.0],
                        [2,3.1],
                        [2,2.8],
                        [2,2.7],
                        [2,3.3],
                        [2,3.4],
                        [2,3.0],
                        [2,3.0],
                        [3,3.1]])
        action2.append([[0,9.8],
                        [3,10.1],
                        [3,10.0],
                        [3,10.0],
                        [3,9.7],
                        [3,10.2],
                        [3,10.0],
                        [3,9.7],
                        [3,10.4],
                        [3,10.0]])
        self.return_values.append(action0)
        self.return_values.append(action1)
        self.return_values.append(action2)
        return
    def reset(self, location=0):
        self.room = location
        return self.room
    def step(self, a):
        draw = random.randint(0,9)
        next_room, rew = self.return_values[a][self.room][draw]
        self.room = next_room
        return next_room, rew
