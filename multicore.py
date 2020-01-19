import tensorflow as tf
import numpy as np
import sys
import multiprocessing as mp
import parmap

np.random.seed(0)
assert tf.executing_eagerly()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    if __name__ == '__main__':
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""

e = 0

n = 1
q = 2
r = 3
b = 4
k = 5
p = 6

N = -1
Q = -2
R = -3
B = -4
K = -5
P = -6

def one_hot(array, n_classes):
  return np.array(tf.keras.backend.one_hot(array, n_classes))

def flip(array):
  return np.flip(array, axis=0)*-1

def init_board():
  board = [[r, k, b, q, n, b, k, r],
          [p, p, p, p, p, p, p, p],
          [e, e, e, e, e, e, e, e],
          [e, e, e, e, e, e, e, e],
          [e, e, e, e, e, e, e, e],
          [e, e, e, e, e, e, e, e],
          [P, P, P, P, P, P, P, P],
          [R, K, B, Q, N, B, K, R]
  ]

  board = np.array(board)

  return board


"""
def weighted_loss(weights):

  def __weighted_loss(logits, labels):
    return tf.keras.backend.mean(tf.keras.backend.square(logits - labels) * weights)

  return __weighted_loss
"""


def weighted_loss(logits, labels):
    weights = labels[1, :, :, :]
    labels = labels[0, :, :, :]
    logits = tf.keras.backend.clip(logits, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # return tf.keras.backend.mean(tf.keras.backend.square(logits - labels) * weights)
    return tf.keras.backend.mean(labels * -tf.math.log(logits) * weights)

inputs = tf.keras.Input(shape=(8, 8, 2 * p + 1), dtype='float32')
# weights = tf.keras.Input(shape=(8, 8, 2*p+1), dtype='float32')

# model = tf.keras.layers.BatchNormalization()(inputs)
model = inputs
# model = tf.keras.backend.one_hot(tf.keras.backend.cast(inputs+p, 'int32'), 2*p+1)
# model = tf.keras.backend.cast(model, 'float32')

for i in range(16):
    cell = tf.keras.layers.Conv2D(filters=2 * p + 1, kernel_size=[1, 1], activation='elu', padding='same')(model)
    cell = tf.keras.layers.Conv2D(filters=2 * p + 1, kernel_size=[3, 3], activation='elu', padding='same')(cell)
    cell = tf.keras.layers.BatchNormalization()(cell)
    model = tf.keras.layers.Concatenate()(list([model, cell]))

model = tf.keras.layers.Conv2D(filters=2 * p + 1, kernel_size=[1, 1], activation='linear', padding='same')(model)
model = tf.keras.layers.Reshape([-1, 2 * p + 1])(model)
model = tf.keras.layers.Softmax(axis=-1)(model)
model = tf.keras.layers.Reshape([8, 8, 2 * p + 1])(model)

model = tf.keras.Model(inputs=inputs, outputs=model)
# model = tf.keras.Model(inputs=[inputs, weights], outputs=model)

model._make_predict_function()

#adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss=weighted_loss)
# model.compile(optimizer='adam', loss=weighted_loss(weights))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer='adam', loss='categorical_crossentropy')

#model._make_predict_function()

def make_move(board, side, randomness=1.): #1 -> ai plays, 0 -> random move
  if side:
    board = flip(board)

  does_king_exist = False
  for i in range(8):
      for j in range(8):
          if board[i, j] == n:
              does_king_exist = True
              break

  if not does_king_exist:
      return 'game_over', np.zeros([8, 8])

  #if np.random.randn(1)*0.5+0.5 < randomness:
  #if np.random.rand(1) <= randomness:
  if np.random.rand(1) < randomness:
    #logits =  model.predict([np.expand_dims(board, 0), np.zeros([1, 8, 8])])
    logits =  model.predict([np.expand_dims(one_hot(board+p, 2*p+1), 0), np.zeros([1, 8, 8])])[0, :, :, :]
    piece, to = decide_move(board, logits)
  else:
    piece, to = random_move(board)


  if piece and to:
    """
    if side:
      print(flip(board), piece, to, sep='\n')
    else:
      print(board, piece, to, sep='\n')
    """

    board[to[0], to[1]] = board[piece[0], piece[1]]
    board[piece[0], piece[1]] = e

    weights_arr = np.zeros([8, 8])
    weights_arr[to[0], to[1]] = 1
    weights_arr[piece[0], piece[1]] = 1

    for j in range(8):
        if board[0, j] == P:
            board[0, j] = Q
        if board[7, j] == p:
            board[7, j] = q

    if side:
      return flip(board), np.flip(weights_arr, axis=0)
    else:
      return board, weights_arr
  else:
    return 'game_over', np.zeros([8, 8])

def decide_move(board, logits):
  biggest_diff = 1 #0
  piece = []
  to = []

  for i in range(8):
    for j in range(8):
      #if abs(board[i, j] - logits[i, j]) > biggest_diff:
      #if -(logits[i, j] - board[i, j]) > biggest_diff:
      if board[i, j] > 0:
        if logits[i, j] / board[i, j] < biggest_diff:
          moves = get_possible_moves(board, [i, j])

          for move in moves:
            copy_board = np.copy(board)
            copy_board[move[0], move[1]] = board[i, j]
            copy_board[i, j] = e

            if not is_check(flip(copy_board), N):
              piece = [i, j]
              to = [move[0], move[1]]

  return piece, to

def decide_move(board, logits):
  biggest_diff = 0
  piece = []
  to = []

  for i in range(8):
    for j in range(8):
      if board[i, j] > 0:
        if 1 - logits[i, j, board[i, j]+p] > biggest_diff:
          moves = get_possible_moves(board, [i, j])

          smallest_diff = 0
          for move in moves:
            if logits[move[0], move[1], board[i, j]+p] > smallest_diff:
              copy_board = np.copy(board)
              copy_board[move[0], move[1]] = board[i, j]
              copy_board[i, j] = e

              if not is_check(flip(copy_board), N):
                piece = [i, j]
                to = [move[0], move[1]]
                biggest_diff =  1 - logits[i, j, board[i, j]+p]
                smallest_diff = logits[move[0], move[1], board[i, j] + p]

  return piece, to

def random_move(board):
  I = list(range(8))
  J = list(range(8))
  np.random.shuffle(I)
  np.random.shuffle(J)
  piece = []
  to = []

  for i in I:
    for j in J:
      if board[i, j] > 0:
        moves = get_possible_moves(board, [i, j])


        np.random.shuffle(moves)

        for move in moves:
          copy_board = np.copy(board)
          copy_board[move[0], move[1]] = board[i, j]
          copy_board[i, j] = e

          if not is_check(flip(copy_board), N):
            piece = [i, j]
            to = [move[0], move[1]]
            break

        if piece and to:
          break

    if piece and to:
      break

  return piece, to

def is_check(board, piece):
  coordinates = []

  for i in range(8):
    for j in range(8):
      if board[i, j] == piece:
        coordinates = [i, j]

  check = False
  for i in range(8):
    for j in range(8):
      if board[i, j] > 0 and not check:
        moves = get_possible_moves(board, [i, j])
        check = coordinates in moves

  return check

def get_possible_moves(board, coordinates): #put together all moves that 'is_legal', are on the board and aren't taking your own pieces
  legal_moves = []
  for i in range(8):
    for j in range(8):
      if (board[coordinates[0], coordinates[1]] > 0 and board[i, j] <= 0) or (board[coordinates[0], coordinates[1]] < 0 and board[i, j] >= 0):
        if is_legal(board, coordinates, [i, j]):
          legal_moves.append([i, j])

  return legal_moves

def is_legal(board, piece, move): #check if move is in a direction that the piece moves andif squares between start and finish are empty
  if board[piece[0], piece[1]] == n: #KINGS
    return abs(piece[0] - move[0]) <= 1 and abs(piece[1] - move[1]) <= 1

  elif board[piece[0], piece[1]] == q: #QUEEN
    if abs(piece[0] - move[0]) == 0:
      if abs(piece[1] - move[1]) >= 2:
        return np.all(board[min([piece[1], move[1]])+1 : max([piece[1], move[1]])] == e)
      else:
        return True
    elif abs(piece[1] - move[1]) == 0:
      if abs(piece[0] - move[0]) >= 2:
        return np.all(board[min([piece[0], move[0]])+1 : max([piece[0], move[0]])] == e)
      else:
        return True
    elif abs(piece[0] - move[0]) == abs(piece[1] - move[1]):

      distance = abs(piece[0] - move[0])
      vertically = np.arange(distance) * (1 if piece[0] < move[0] else -1)
      horizontally = np.arange(distance) * (1 if piece[1] < move[1] else -1)

      empty = True
      for i in range(1, distance):
        if board[piece[0]+vertically[i], piece[1]+horizontally[i]] != e:
          empty = False

      return empty
    else:
      return False

  elif board[piece[0], piece[1]] == r: #ROOK
    if abs(piece[0] - move[0]) == 0:
      if abs(piece[1] - move[1]) >= 2:
        return np.all(board[min([piece[1], move[1]])+1 : max([piece[1], move[1]])] == e)
      else:
        return True
    elif abs(piece[1] - move[1]) == 0:
      if abs(piece[0] - move[0]) >= 2:
        return np.all(board[min([piece[0], move[0]])+1 : max([piece[0], move[0]])] == e)
      else:
        return True
    else:
      return False

  elif board[piece[0], piece[1]] == b: #BISHOP
    if abs(piece[0] - move[0]) == abs(piece[1] - move[1]):

      distance = abs(piece[0] - move[0])
      vertically = np.arange(distance) * (1 if piece[0] < move[0] else -1)
      horizontally = np.arange(distance) * (1 if piece[1] < move[1] else -1)

      empty = True
      for i in range(1, distance):
        if board[piece[0]+vertically[i], piece[1]+horizontally[i]] != e:
          empty = False

      return empty

    else:
      return False

  elif board[piece[0], piece[1]] == k: #KNIGHT
    return (abs(piece[0] - move[0]) == 2 and abs(piece[1] - move[1]) == 1) or (abs(piece[0] - move[0]) == 1 and abs(piece[1] - move[1]) == 2)

  elif board[piece[0], piece[1]] == p: #PAWN
    return ((move[0] - piece[0]) == 1 and abs(piece[1] - move[1]) == 0 and board[move[0], move[1]] == e) or ((move[0] - piece[0]) == 1 and abs(piece[1] - move[1]) == 1 and board[move[0], move[1]] < 0) or (piece[1] == move[1] and piece[0] == 1 and move[0] == 3 and board[2, move[1]] == e and board[3, move[1]] == e)

  else:
    print('u fucked up')

n_steps = 20
start = 0#44
games_to_play = 100
max_depth = 100
epochs = 2
batch_size = 512
n_cores = mp.cpu_count()


checkpoint_path = "training_7/cp.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

if False:
    load_path = "training_6/cp.ckpt"
    model.load_weights(load_path)

def play_games(info):
    step, memory = info

    if start != step:
        model.load_weights(checkpoint_path)

    for game in range(games_to_play//n_cores):
        game_memory = []
        board = init_board()
        #print("NEW GAMEEEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", game)

        for depth in range(max_depth):
            move, weights_arr = make_move(np.copy(board), False, step / n_steps)#step / n_steps
            if not (type(move) is np.ndarray):
                win = 'black'
                break
            else:
                game_memory.append([board, move, weights_arr])
                board = np.copy(move)

            move, weights_arr = make_move(np.copy(board), True, step / n_steps)#step / n_steps
            if not (type(move) is np.ndarray):
                win = 'white'
                break
            else:
                game_memory.append([flip(board), flip(move), np.flip(weights_arr, axis=0)])
                # board = np.copy(flip(move))
                board = np.copy(move)

        else:
            win = 'white' if np.sum(board > 0) > np.sum(board < 0) else 'black'

        memory.append([win, game_memory])

if __name__ == '__main__':
    #white_wr = 0
    #black_wr = 0
    for step in range(start, n_steps):
        print('STEP NUMBER:', step)

        replay_memory = mp.Manager().list()#[]

        #pool = mp.Pool()
        #pool.map(play_games, [replay_memory] * n_cores)
        #pool.close()
        #pool.join()
        """
        processes = []
        for i in range(0, n_cores):
            p = mp.Process(target=play_games, args=(step,))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()
        """
        parmap.map(play_games, [(step, replay_memory)] * n_cores)



        x = []
        w = []
        y = []

        """
        for game in replay_memory:
            result, moves = game
            if result == 'white':
                white_wr += 1
            if result == 'black':
                black_wr += 1
        print(white_wr, black_wr)
        sys.exit()
        """

        for game in replay_memory:
            result, moves = game

            for iterator, data in enumerate(moves):
                board, move, weights_arr = data

                x.append(one_hot(board+p, 2*p+1))
                #x.append(board)
                w.append(np.tile(np.expand_dims(weights_arr, -1), [1, 1, 13]))
                #w.append(weights_arr)
                if (result == 'white' and iterator % 2 == 0) or (result == 'black' and iterator % 2 == 1):
                    y.append(one_hot(move+p, 2*p+1))
                    #y.append(move)
                else:
                    y.append(one_hot(board+p, 2*p+1))
                    #y.append(board)

        x = np.array(x)
        w = np.array(w)
        y = np.array(y)
        yw = np.stack([y, w], 1)

        whole_batches = x.shape[0]//batch_size
        x = x[:whole_batches*batch_size]
        yw = yw[:whole_batches*batch_size]

        # model.fit(x=[x, w], y=y, epochs = epochs, callbacks=[cp_callback])
        # model.fit(x=x, y=y, epochs = epochs, callbacks=[cp_callback])
        model.fit(x=x, y=yw, epochs=epochs, callbacks=[cp_callback], verbose=2, batch_size=batch_size)
