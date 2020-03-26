import tensorflow as tf
import numpy as np

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return tf.keras.backend.one_hot(tf.keras.backend.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return tf.keras.layers.Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))

model = tf.keras.Sequential([
    tf.keras.layers.Input([3, 3]),
    #tf.keras.layers.Reshape([3, 3, 1]),
    OneHot(3, 1024),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(9, 3, padding='valid'),
    #global average pooling
    tf.keras.layers.Flatten(),
    tf.keras.layers.Softmax()
])

model = tf.keras.Sequential([
    tf.keras.layers.Input([3, 3]),
    OneHot(3, 1024),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(36, activation='relu'),
    tf.keras.layers.Dense(36, activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Softmax()
])

model.compile(tf.keras.optimizers.Adam(0.001), 'sparse_categorical_crossentropy')
model.summary()

checkpoint_path = "training_0/cp.ckpt"

def network_move(board):
    logits = model.predict(board.reshape(1, 3, 3)+1)[0]

    max = 0
    logit = None
    for i in range(9):
        if logits[i] >= max and board[i//3, i%3] == 0:
            max = logits[i]
            logit = i

    board[logit//3, logit%3] = 1

    return board, logit

def random_move(board):
    logits = np.arange(9)
    np.random.shuffle(logits)

    logit = None
    for i in logits:
        if board[i//3, i%3] == 0:
            logit = i
            break

    board[logit // 3, logit % 3] = 1

    return board, logit

def make_move(board, random):
    if np.random.rand() < random:
        return network_move(board)
    else:
        return random_move(board)

def is_game_over(board):
    if np.any(np.sum(board, axis=0) == 3) or np.any(np.sum(board, axis=0) == -3) or np.any(np.sum(board, axis=1) == 3) or np.any(np.sum(board, axis=1) == -3):
        return True
    if board[0, 0] + board[1, 1] + board[2, 2] in [-3, 3]:
        return True
    return False

just_play = True
if not just_play:
    num_steps = 100
    num_games = 2000
    num_epochs = 100

    for step in range(num_steps):
        print('STEP:', step)

        x = []
        y = []
        for game in range(num_games):
            board = np.zeros([3, 3])
            game_memory = []
            win = None

            for i in range(9):
                move, pred = make_move(np.copy(board), step/num_steps)
                game_memory.append([board,  pred])
                board = move
                if is_game_over(board):
                    win = 1
                    break
                if np.sum(np.abs(board)) == 9:
                    break

                move, pred = make_move(np.copy(board) * -1, step/num_steps)
                move = move * -1
                game_memory.append([board * -1, pred])
                board = move
                if is_game_over(board):
                    win = -1
                    break
                if np.sum(np.abs(board)) == 9:
                    break

            for num, tpl in enumerate(game_memory):
                i, j = tpl
                if (win != -1 and num % 2 == 0) or (win != 1 and num % 2 == 1):

                    x.append(i)
                    y.append(j)

        x = np.stack(x)
        y = np.stack(y)

        print(x.shape, y.shape)

        model.fit(x+1, y, batch_size=1024, epochs=num_epochs, verbose=0)
        model.evaluate(x, y)

        model.save_weights(checkpoint_path)
else:
    model.load_weights(checkpoint_path)

a = np.zeros([3, 3])
while True:
    print(a)
    logit = int(input())
    a[logit//3, logit%3] = -1
    print(is_game_over(a))
    a, _ = make_move(a, 1)
    print(is_game_over(a))