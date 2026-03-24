# A coinche simulator to practice RL in Jax

## Datastructures

There are currently three main structures. Each of them has a convenient representation, and a tensor representation. Most of the time, the *convenient* representation must be used to keep the code clear. Note that both representations support batching for vectorized computations.

### Cards
`Cards` are represented as a tuple `(Int, Int)`, the first integer denotes the *Suit* $\in \\{0,\ldots,3\\}$ and the second integer represents the *Rank* $\in \\{0,11\\}$. The tensor representation is a vector in $\\{0,1\\}^{12}$ where the suit is one-hot encoded on the $4$ first bits and the ranks is one-hot encoded on the $8$ last bits.

The ranks are assumed to be in decreasing order (common way to sort the cards). In the *coinche*, this means that *most of the time* Ace is ranked $0$ while the $7$ is ranked $7$. However, in this game, the card order depends on the suit (weither it is a *trump* or not). For instance, in the trump suit, *Jack* becomes ranked first. In our representation, we are assuming that this conversion is already made and that all suit are ranked properly, i.e. a card will always beat cards with higher ranks.
