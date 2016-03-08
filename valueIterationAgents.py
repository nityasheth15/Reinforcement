# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your tempValue iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.stateActionMap = {}
        
        for eachState in self.mdp.getStates():
            self.stateActionMap[eachState]=None
        
        for iteration in range (iterations):
            valuesForEachIteration = self.values.copy()
            for eachState in self.mdp.getStates():
                if self.mdp.isTerminal(eachState):
                    continue
                value = float("-inf")
                for eachAction in self.mdp.getPossibleActions(eachState):
                    tempValue=0
                    for eachPossibleState in self.mdp.getTransitionStatesAndProbs(eachState,eachAction):
                        nextPossibleState = eachPossibleState[0]
                        nextStateprobabliity = eachPossibleState[1]
                        tempValue += nextStateprobabliity*(self.mdp.getReward(eachState,eachAction,nextPossibleState) + self.discount*self.getValue(nextPossibleState))
                    if(tempValue>value):
                        valuesForEachIteration[eachState] = tempValue
                        self.stateActionMap[eachState] = eachAction
                        value = tempValue
            self.values= valuesForEachIteration.copy()

    def getValue(self, state):
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        qValue = 0
        possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for eachPossibleState in possibleNextStates:
            qValue += eachPossibleState[1] * (self.mdp.getReward(state, action, eachPossibleState[0]) + self.discount * self.getValue(eachPossibleState[0]))
        return qValue
        

    def computeActionFromValues(self, state):
        return self.stateActionMap[state]
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
