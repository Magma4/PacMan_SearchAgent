# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Define a stack variable to store state and actions
    state_stack = util.Stack()
    # Define a set variable to store visited nodes
    visited_nodes = set()

    # Get start state and addto stack
    start_state = problem.getStartState()
    state_stack.push((start_state, []))

    # Checl if state statck is empty
    while not state_stack.isEmpty():
        # Remove current node and actions from stack
        current_node, actions = state_stack.pop()

        # Check if current node is not in visited nodes
        if current_node not in visited_nodes:
            # Add current node to visited nodes
            visited_nodes.add(current_node)

            # Check if current node is a goal state
            if problem.isGoalState(current_node):
                # Return actions
                return actions

            # Get successors of current node
            for successor, action, cost in problem.getSuccessors(current_node):
                # Check if successor is not in visited nodes
                if successor not in visited_nodes:
                    # Add action to actions
                    current_actions = actions + [action]
                    # Add successor and actions to stack
                    state_stack.push((successor, current_actions))

    # Return empty list if no path is found
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # Define a queue variable to store state and actions
    path = util.Queue()
    # Define a set variable to store visited nodes
    visited_nodes = set()

    # Get start state and add to queue
    start_state = problem.getStartState()
    # Add start state and actions to queue
    path.push((start_state, []))

    # Check if path is empty
    while not path.isEmpty():
        # Remove current node and actions from queue
        (current_node, actions) = path.pop()

        # Check if current node is not in visited nodes
        if current_node not in visited_nodes:
            # Add current node to visited nodes
            visited_nodes.add(current_node)

            # Check if current node is a goal state
            if problem.isGoalState(current_node):
                # Return actions
                return actions

            # Get successors of current node
            for successor, action, cost in problem.getSuccessors(current_node):
                # Check if successor is not in visited nodes
                if successor not in visited_nodes:
                    # Add action to actions
                    current_actions = actions + [action]
                    # Add successor and actions to queue
                    path.push((successor, current_actions))
    # Return empty list if no path is found
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    path = util.PriorityQueue()
    visited_node = set()
    current_cost = 0
    path.push((start_state, [], 0), current_cost)

    while not path.isEmpty():
        current_node, actions, total_cost = path.pop()

        if current_node not in visited_node:
            visited_node.add(current_node)

            if problem.isGoalState(current_node):
                return actions

            for successor, action, cost in problem.getSuccessors(current_node):
                if successor not in visited_node:
                    new_actions = actions + [action]
                    new_cost = total_cost + cost
                    path.push((successor, new_actions, new_cost), new_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    start_state = problem.getStartState()
    path = util.PriorityQueue()
    visited_node = set()
    cost = 0
    heuristic_cost = heuristic(start_state, problem)
    true_cost = cost + heuristic_cost
    path.push((start_state, [], 0), true_cost)

    while not path.isEmpty():
        current_node, actions, total_cost = path.pop()

        if current_node not in visited_node:
            visited_node.add(current_node)

            if problem.isGoalState(current_node):
                return actions

            for successor, action, cost in problem.getSuccessors(current_node):
                if successor not in visited_node:
                    new_actions = actions + [action]
                    new_cost = total_cost + cost
                    new_heuristic_cost = heuristic(successor, problem)
                    new_true_cost = new_cost + new_heuristic_cost
                    path.push((successor, new_actions, new_cost), new_true_cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
