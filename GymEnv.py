import gym

from dataModel import BurgleBros, read_config
from tiles import Floor, Stairs, vertical_wall_configs, horizontal_wall_configs, DIRECTIONS, ENCODING
from units import Player, Guard


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2):
    super(CustomEnv, self).__init__()

    self.config = read_config('./config.ini')
    self.game = BurgleBros(**self.config)


  def get_observation(self):
    # get observation data for the agents to make choices
    player = self.game.players[self.game.current_player]
    # tell tile unit moved
    y, x, floor = player.get_location()
    current_tile = self.game.floors[floor].get_tiles()[y][x]

    # TODO: the 'observation' should be the moves available at that time?
    available_moves = current_tile.get_moves()
    # TODO: mot sure if 'quit' counts or any of the current 'safe' moves
    # TODO: safe moves have a check after they have been picked for if the tile is a safe tile
    print("\nq to quit, 'add' to add dice to safe, 'roll' to attempt to open the safe")

    observation = available_moves
    return observation

  def get_reward(self):

    # loop over players
    for player in self.game.players:
      # TODO: took a guess on how to implement reward based off what I can get from current game state
      if player.sneak_tokens > 0 and not self.game.finished:
        return 100

    return -1

  def step(self, p1_action, p2_action):
    # TODO: I do not know how to incorporate 'p1_action' and 'p2_action'
    # Execute one time step within the environment
    # show the game board
    self.game.print_board()

    # prompt current player for move
    desired_move, player, current_tile = self.prompt_action()
    self.execute_action(desired_move, player)
    player = self.game.players[self.game.current_player]

    # TODO: this part about checking the actions might need to be moved out?
    # TODO: or put in another while loop?
    if player.get_actions_left() == 0:
      # if no moves left, move guard

      # select next player
      self.game.current_player = (self.game.current_player + 1) % self.game.num_players
      player = self.game.players[self.game.current_player]
      self.game.players[self.game.current_player].set_action_count(player.get_actions_per_turn())

    self.move_guard(player)

  def reset(self):
    # Reset the state of the environment to an initial state
    # TODO: should reset the whole game over
    # TODO: not sure if the 'reset' function means reset whole game or the turn
    self.game = BurgleBros(**self.config)

    return self.get_observation()

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    # TODO: I don't think I need anything else?
    return self.game.print_board()

  def move_guard(self, player):

    _, _, floor = player.get_location()
    if self.game.use_guards:
      guard: Guard = self.game.guards[floor]
      num_actions = guard.calc_speed()
      guard_path = [guard.get_location()]
      for i in range(num_actions):
        new_loc = guard.move()
        guard_path.append(new_loc)
        self.game.floors[floor].get_tiles()[new_loc[0]][new_loc[1]].unit_enters(guard)
        for p in self.game.players:
          if p.get_location() == (*new_loc, floor):
            count = p.remove_sneak_token()
            if count < 0:
              self.game.finished = -1
      print("Guard took the following path: ", guard_path)

  def prompt_action(self):
    player = self.game.players[self.game.current_player]
    # tell tile unit moved
    y, x, floor = player.get_location()
    current_tile = self.game.floors[floor].get_tiles()[y][x]

    available_moves = current_tile.get_moves()
    print("\nq to quit, 'add' to add dice to safe, 'roll' to attempt to open the safe")
    print("Direction commands: ", DIRECTIONS, '\n')
    print('available for your tile:', available_moves)

    desired_move = ""
    while desired_move not in available_moves:
      desired_move = input("Please enter your move:")
      if desired_move not in available_moves:
        print("Invalid move")
        desired_move = ""

    return desired_move, player, current_tile

  def execute_action(self, desired_move, player):
    # tell tile unit moved
    y, x, floor = player.get_location()
    current_tile = self.game.floors[floor].get_tiles()[y][x]

    # todo: add 'end turn' option that sets actions_left to 0
    if desired_move == 'q':
      self.game.finished = -1
      return
    if desired_move == 'x':
      player.set_action_count(0)
    elif desired_move == 'add':
      current_tile.add_dice()
      player.set_action_count(player.get_actions_left() - 1)
    elif desired_move == 'roll':
      rolls = current_tile.roll_dice()
      print("Rolled:", rolls)
      self.game.floors[floor].add_dice_rolls(rolls)
      player.set_action_count(player.get_actions_left() - 1)
    elif desired_move == 'w':
      player.move_unit(y - 1, x, floor)
    elif desired_move == 'a':
      player.move_unit(y, x - 1, floor)
    elif desired_move == 's':
      player.move_unit(y + 1, x, floor)
    elif desired_move == 'd':
      player.move_unit(y, x + 1, floor)
    elif desired_move == 'e':
      safe_states = [f.is_safe_open() for f in self.game.floors]
      if all(safe_states) and floor == self.game.num_floors - 1:
        self.game.finished = 1
      elif floor == self.game.num_floors - 1:
        print("Cannot go up until all safes are opened: ", safe_states)
        return
      else:
        player.move_unit(y, x, floor + 1)
    elif desired_move == 'z':
      player.move_unit(y, x, floor - 1)
    else:
      print("Invalid move ", desired_move)

