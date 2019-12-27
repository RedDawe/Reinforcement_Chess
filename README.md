# Reinforcement_Chess

Dumber, retarded brother of AlphaZero, who got beat up with a baseball bat in the womb.

# Simplifications of the Rules

- during training there are no draws (stalemate is a loss for the one that cannot move and everything else is ignored and after a number of moves exceeds a specified threshold the player with more material wins)

- en passant (completely)

- castling (also not allowed)

- pawn at the end of the board gets automatically traded for a QUEEN (sometimes u would like another piece)

Some of these rules (castling) are sometimes not allowed for computers as the games tend to be more interesting in that case, some of the rules (en passant, queen) just don't happen often enough to be bothered with and stalemate classification is only effective in training, which may or may not have a positive effect on the actuall games.

# Some more complications

- As always computing power is scarce. In this case tho the main problem isn't the training of the network but playing the training games. That is mainly caused by the fact that we use tensorflow and gpu to train the network, but playing the games is done by only single core python/numpy. Rewriting this part of the code in a compiled language like C++ and with multi core in mind, this issue would be much less of an issue (intended).
