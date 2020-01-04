# Reinforcement_Chess

Dumber, retarded brother of AlphaZero, who got beat up with a baseball bat in the womb.

# Simplifications of the Rules

- during training there are no draws (stalemate is a loss for the one that cannot move and everything else is ignored and after a number of moves exceeds a specified threshold the player with more material wins)

- en passant (completely)

- castling (also not allowed)

- pawn at the end of the board gets automatically traded for a QUEEN (sometimes u would like another piece)

Some of these rules (castling) are sometimes not allowed for computers as the games tend to be more interesting in that case, some of the rules (en passant, queen) just don't happen often enough to be bothered with and stalemate classification is only effective in training, which may or may not have a positive effect on the actuall games.

# Stockfish cheat

To adress computational scarcity, in this implementation we don't start with random moves but moves genereated by stockfish. It takes longer to generate the moves, but network requires less iterations to start learning something decent. At the end the network still self plays tho, so it is not capping its upside. It is not tabula rasa, but it still is a neural net playing chess.

# Conclusion

The network would need more training, but the completely random version did learn to reliably beat random moves if game length was capped at 20 moves (100 during training). This is not that terrible considering it had only played 2000 games out of the 72000+ possible at that moment. This whole thing is done without any special preprocessing except one_hot piece representation. DeepMing used many other features for Go, but I couldn't find their preprocessing for chess. The code could use paralisation during playing the training games.
