import jax.numpy as jnp
import jax

ACE, TEN, K, Q, J, NINE, EIGHT, SEVEN = 0,1,2,3,4,5,6,7
to_trump = jnp.array([J,NINE,ACE,TEN,K,Q,EIGHT,SEVEN])
J, NINE, ACE, TEN, K, Q,  EIGHT, SEVEN = 0,1,2,3,4,5,6,7
to_color = jnp.array([ACE,TEN,K,Q,J, NINE, EIGHT,SEVEN])


ALL_TRUMP = -1
NO_TRUMP = -2

class Hand:
    def __init__(self, colors): # colors: np.array (4,8) Bool sorted sans at
        # sorted decreasingly: Ace 10 K Q J 9 8 7
        #                      0   1  2 3 4 5 6 7
        self.hand = colors

        self.trump = -1 # -1 = sans at, -2 = toutat


    def set_trump (self, color_idx):
        """
       Reorders the colors (if it was sans at, it becomes trump, and vice versa) 
        """
        self.hand[color_idx] = self.hand[color_idx][to_trump]


    def unset_trump (self, color_idx):
        self.hand[color_idx] = self.hand[color_idx][to_color]




class Trick:
    def __init__(self, player_index, first_card): #card = [color, rank]
        color, rank = first_card

        self.color = color
        self.cards = [first_card]
        self.best_card = first_card

    def add (self, player_index, trump_color, new_card):
        color, rank = new_card
        



            
