
#import pygame

#from pygame import mixer

#from pygame.locals import *

from collections import deque

import sys

import random

import numpy as np

import math

import tensorflow as tf

import matplotlib.pyplot as plt

import pickle as pkl

import csv

#Save ReplayBuffer, save state,action, next_state,

#mixer.init()

MAX_SCORE = 0



"""Code"""

class Vector(tuple):

   """Magic Methods"""

   """Vector will be classified as a tuple"""

   def __add__(self, other):

      return Vector(x + z for x,z in zip(self, other))

   def _radd__(self, other):

      return Vector(z + x for x,z, in zip(self, other))

   def __sub__(self, other):

      return Vector(x-z for x,z in zip(self, other))

   def __rsub__(self, other):

      return Vector(z-x for x,z in zip(self, other))

   def __mul__(self, sf):

      return Vector(x * sf for x in self)

   def __rmul__(self, sf):

      return Vector(sf * x for x in self)

   def __neg__(self):

      return -1 * self

"""Music Files"""

#pygame.mixer.music.load('TetrisMusic.mp3')

#pygame.mixer.music.play(-1)

#Music will play indefinitely

FPS = 30 # 10 frames per second. clock.tick(10) - prevents framerate>10

snake_initial_length = 3

snake_initial_speed = 3 #3 squares per second

world_size = Vector((20,20))

block_size = 24

world_in_blocks = world_size * block_size





"""Colours - RGB"""

#Black

background_colour = 0,0,0

#Green

snake_colour = 0,255,0

#Red

food_colour = 255,0,0

death_colour = 255,0,0

#White

text_colour = 255,255,255



"""Directions"""

direction_UP = Vector((0,-1))

direction_DOWN = Vector((0,1))

direction_LEFT = Vector((-1,0))

direction_RIGHT = Vector((1,0))



#Keep relevant keys for direction in dictionary

#(wasd, up, left, right, down arrows)

"""https://www.pygame.org/docs/ref/key.html"""

#keys_for_Direction = {
#   K_w: direction_UP,    K_UP: direction_UP,
#   K_a: direction_LEFT,  K_LEFT: direction_LEFT,
#   K_s: direction_DOWN,  K_DOWN: direction_DOWN,
#   K_d: direction_RIGHT, K_RIGHT: direction_RIGHT
#
#  }







class Snake():

   def __init__(self,start_Point, snake_initial_length):

      self.time = FPS

      """Body of snake - will increase length of snake by 1. range = 0,1 if = 2"""

      self.direction = direction_RIGHT

      self.body = deque([start_Point - self.direction * i for i in range(snake_initial_length)])

      #self.world = Rect((0,0), world_size)

      """Define snake in deque

         Starts off in centre of grid - 2 is initial length, so 1 head, 1 body

         """

   def __iter__(self):

      return iter(self.body)

   def __len__(self):

      return len(self.body)

   def head_of_snake(self):

      """This will return position 0 of queue = head"""

      return self.body[0]

   def movement_of_snake(self, action):

      """Will action also be used for action of agent?"""

      if self.direction != -action:

         self.direction = action

      """Action occurs, snake head moves according to that

         only if action taken is not equal to the negative direction

         e.g. going up when currently going down

              going left when going right currently"""

   def intersecting_snake(self):

      #Future: only head is checked to remain rather than checking repetition of everything

      self_intersect = set()

      for i in range(len(self.body)):

         self_intersect.add(self.body[i])

      if len(self_intersect) != len(self.body):

         return True

      return False



      """Put all items in the queue into a set. If set length != length of queue

         Then snake is self-intersecting"""







class grid():

   def __init__(self):

      """Good practise to move to local variables"""

      self.world_size = world_size

      self.snake_initial_length = snake_initial_length

      self.reward = 0

      

      #self.clock = pygame.time.Clock()

      #sans is default font

      #self.font = pygame.font.SysFont(None, 30)

      #self.world = Rect((0,0), world_size)

      self.reset_game()

      self.state = self.state_Q_table()

   def reset_game(self):

      self.game_over = False

      self.game_iteration = 0

      self.next_direction = direction_RIGHT

      self.score = 0

      x = random.randint(4,19)
      y = random.randint(0,19)

      self.snake = Snake((x,y),snake_initial_length)

      self.food = set()

      self.add_food()

      self.state = self.state_Q_table()

      #self.game_over = tf.Tensor(game_over, dtype = tf.uint8)

   def add_food(self):

      while True:

         food_coordX = random.randint(0, world_size[0]-1)

         food_coordY = random.randint(0, world_size[1]-1)

         food_generated = Vector((food_coordX, food_coordY))

         if food_generated not in self.food and food_generated not in self.snake.body:

            self.food.add(food_generated)

            return food_generated

         else:

            #print(f"Apple maybe did spawn at coordinates but who tf knows: ({food_coordX}, {food_coordY})")

            self.add_food()



            

         

   def add_food_to_Snake(self, MAX_SCORE):

      self.snake.body.appendleft(self.snake.head_of_snake() + self.snake.direction)

      if self.snake.head_of_snake() in self.food:

         self.food.remove(self.snake.head_of_snake())

         self.add_food()

         self.reward = 10

         #Increase Snake length by 1

         self.score = self.score + 1

         if self.score >= MAX_SCORE:

            MAX_SCORE = self.score

      else:

         self.snake.body.pop()

   def state_Q_table(self):

      #Initialise Q-table

      Q_table = np.zeros(self.world_size)

      for segment in range(len(self.snake.body)):

        #Keys in Q-table array

         Q_table[segment] = 1

      food = list(self.food)

      num_food = len(food)

      Q_table[food[self.score % num_food]] = 2

      return np.array(Q_table)

   

   def array_my_food(self):

      food_array = []

      for x in self.food:

         food_array.append(x)

      return food_array[self.score]

      

   def distance_to_food(self,score):

        #Euclidean distance - Pythagoras

         head_array = np.array(self.snake.head_of_snake())

         food_list = np.array(self.array_my_food())

         #print((self.snake.head_of_snake()[0] - self.array_my_food()[0])^2 + (self.snake.head_of_snake()[1] - self.array_my_food()[1])^2)

        #Call position in set that corresponds with turn

         return np.linalg.norm(head_array-food_list)

         #return math.sqrt((self.snake.head_of_snake()[0] - self.add_food()[0])^2 + (self.snake.head_of_snake()[1] - self.add_food()[1])^2)

   def is_snake_dead(self):

      #Snake dead either touching borders of game or self-intersecting

      if (self.snake.head_of_snake()[0] > world_size[0] or self.snake.head_of_snake()[1] > world_size[1]):

         self.game_over = True

      elif (self.snake.head_of_snake()[0] < 0 or self.snake.head_of_snake()[1] < 0):

         self.game_over = True

      elif self.snake.intersecting_snake() == True:

         self.game_over = True

      else:

         self.game_over = False

   def update_frame(self):

      self.snake.movement_of_snake(self.next_direction)

      #Update snake direction

      self.add_food_to_Snake(MAX_SCORE)

      if self.is_snake_dead() == True:

         #self.draw_death()

         self.reset_game()

      #self.clock.tick(FPS)     

   def play(self, action):

      while True:

        # Up = L or R

        # Down = L or R

        # Left = U or D

        # Right = U or D

         if action == 0:

            self.snake.movement_of_snake(direction_UP)

         elif action == 1:

            self.snake.movement_of_snake(direction_DOWN)

         elif action == 2:

            self.snake.movement_of_snake(direction_LEFT)

         elif action == 3:

            self.snake.movement_of_snake(direction_RIGHT)

        

         self.game_iteration = self.game_iteration + 1

         new_distance = self.distance_to_food(self.score)

         if self.game_over == False and self.game_iteration < (100 * len(self.snake)):

            old_distance = self.distance_to_food(self.score)

            self.update_frame()

            #self.draw_game_playing() #AI needs no actual imagery

         else:

            self.game_over = True  

            if old_distance>=new_distance:

                self.reward = -10

            else:

                self.reward = -1



        #pygame.display.flip()

         return self.reward, self.game_over, self.state_Q_table()

NUM_OF_EPISODES = 50

discount_factor = 0.99

memory_value = 10000

class DeepQNetwork(tf.keras.Model):

    def __init__(self, input_world):

      super(DeepQNetwork, self).__init__()

      self.input_world = tf.keras.Input(shape=(input_world[0], input_world[1]))

      self.input_world = tf.keras.Input(shape=(input_world[0], input_world[1]))

      #FC - fully connected layer -> each node connected to all other nodes in layer

      self.input_world = tf.keras.Input(shape = (input_world[0],input_world[1]))

      self.fc1 = tf.keras.layers.Dense(128, activation='relu')

      self.fc2 = tf.keras.layers.Dense(128, activation='relu')

      self.output_layer = tf.keras.layers.Dense(4, activation='linear')

    def call(self,inputs):

      x = self.fc1(inputs)

      x = self.fc2(x)

      return self.output_layer(x)

capacity = 10_000

batch_size = 24

current_model = DeepQNetwork(world_size)

class ReplayBuffer(tf.Module):

    def __init__(self, capacity):

        self.capacity = capacity

        self.memory_buffer = []



    def add_batch(self, items):

        if len(self.memory_buffer) >= self.capacity:

            self.memory_buffer.pop(0)

        self.memory_buffer.append(items)



    def sample(self, batch_size):

        if len(self.memory_buffer) < batch_size:

            return []

        else:

            return random.sample(self.memory_buffer, batch_size)



class SnakeModel():

    def __init__(self, input_world, state, action, reward, next_state, game_over):

      super(SnakeModel, self).__init__()

      self.state = np.array([state])  # Use provided state

      self.action = np.array([action])  # Use provided action

      self.reward = np.array([reward])  # Use provided reward

      self.next_state = np.array([next_state])  # Use provided next_state

      self.game_over = np.array([[game_over]], dtype=np.uint8)  # Use provided game_over

      self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

      """

      self.state = tf.Tensor(state, dtype = tf.float32)

      self.action = tf.Tensor(action, dtype = tf.float32)

      self.reward = tf.Tensor(reward, dtype = tf.float32)

      self.next_state = tf.Tensor(next_state, dtype = tf.float32)

      self.game_over = tf.Tensor(game_over, dtype =tf.uint8)"""

      self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def train_DQN(self, replay_buffer, batch_size):

      if(len(replay_buffer.memory_buffer) < batch_size):

         #Only start when greater than batch_size

         return

      q_values =  current_model(np.array(state))

      next_q_values = current_model(np.array(next_state))

      target_q_values = q_values.numpy()

      max_next_q_values = np.max(next_q_values, axis=1)

      print("target_q_values shape:", target_q_values.shape)

      #print("self.action shape:", self.action.shape)

      print("self.reward shape:", self.reward.shape)

      #print("self.game_over shape:", self.game_over.shape)

      #print(reward)

      for i in range(batch_size):

          action_index = int(action)

          target_q_values = target_q_values.astype(int)

          self.reward = self.reward.astype(int)

          self.game_over = self.game_over.astype(int)

          for i in range(batch_size):

                target_q_values[i][action_index] = self.reward[i] + (1 - self.game_over[i]) * discount_factor * max_next_q_values[i]

      with tf.GradientTape() as tape:

         #qvalues

         q_value = np.array(grid().reset_game())

         loss = keras.losses.mean_squared_error(target_q_values,q_value)

         #loss function



      

      trainable_vars = self.trainable_variables

      gradients = tape.gradient(loss, trainable_vars)

      self.optimizer.apply_gradients(zip(gradients, trainable_vars))





next_model = DeepQNetwork(grid().world_size)

replay_buffer = ReplayBuffer(capacity=10000)

epsilon = 0.5

epsilon_decay_value = 0.99





#episodes

episode_vs_score = {"Episode Number:": [], "Score:": []}                            

for episode in range(NUM_OF_EPISODES):

    episode_reward = 0

    epsilon = epsilon * epsilon_decay_value

    state = grid().state_Q_table()

    action = 0



    while not grid().game_over:

        if random.random() > epsilon_decay_value:

            q_values = current_model(np.array(state))

            action = np.argmax(q_values[0])

        else:

            action = random.randint(0, 3)



        reward, game_over, next_state = grid().play(action)

        episode_reward += reward



        replay_buffer.add_batch((state, action, reward, next_state, game_over))

        batch = replay_buffer.sample(batch_size)



        if len(batch) == 0:

            continue



        state_batch, action_batch, reward_batch, next_state_batch, game_over_batch = zip(*batch)

    for i in range(batch_size):

        snake_model = SnakeModel(world_size, state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], game_over_batch[i])

        snake_model.train_DQN(replay_buffer, batch_size)

        state = next_state



    episode_vs_score["Episode Number:"].append(episode + 1)

    episode_vs_score["Score:"].append(episode_reward)



    current_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))



    if (episode + 1) % 1000 == 0:

        print(f"Episode {episode + 1}: Score = {episode_reward}")



print("Maximum Score: ", MAX_SCORE)



#Will plot graph of episode no. vs score

plt.plot(episode_vs_score["Episode Number:"],episode_vs_score["Score:"])

plt.xlabel("Episode Number")

plt.ylabel("Score")

plt.savefig("Ep_vs_Score_Snake_"+str(epsilon)+"_.png")

plt.show()





"""Saving Files"""

# Method1:

snakeTensors = [state_batch, action_batch, reward_batch, next_state_batch, game_over_batch]
with open("replay_buffer.pkl", "wb") as replay_file:
    pkl.dump(replay_buffer, replay_file)
with open("snakeTensors.pkl", "wb") as snakeTensors_file:
    pkl.dump(snakeTensors, snakeTensors_file)





# Method2:



# model_config = model.to.json()

# with open("model_config.json", "w") as json_file:

#    json_file.write(model_config)



# Method:3

"""print("Save")

np.save("q_values.npy", q_values)

np.save("next_q_values.npy",next_q_values)

np.save("target_q_values.npy",target_q_values)

np.save("max_next_q_values.npy",max_next_q_values)

np.save("replay_buffer.npy",replay_buffer)"""



#  Method 4:



#  data = {"Episode": [], "Reward": []}

 

#  # Append data for each episode

#  data["Episode"].append(episode)

#  data["Reward"].append(episode_reward)

 

#  with open("training_data.csv", "w", newline="") as csv_file:

#      csv_writer = csv.DictWriter(csv_file, fieldnames=data.keys())

#      csv_writer.writeheader()

#      csv_writer.writerows(data)



	
