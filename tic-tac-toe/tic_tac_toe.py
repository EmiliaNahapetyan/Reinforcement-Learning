from state import get_all_states
from player import RLPlayer, HumanPlayer
from judge import Judge

# Get all possible board configurations
all_states = get_all_states(rows=3, columns=3)

# region Functions

def train(epochs: int, print_every_n: int = 500):
    # region Summary
    """
    Train 2 RL players
    :param epochs: number of epochs for training
    :param print_every_n: number of epochs to print the intermediate win rate
    """
    # endregion Summary

    # region Body

    # Create 2 RL players with ε = 0.01 exploring probability
    player1 = RLPlayer(all_states, epsilon=0.01)
    player2 = RLPlayer(all_states, epsilon=0.01)

    # Create a judge to organize the game
    judge = Judge(player1, player2)

    # Set the initial win rate of both players to 0
    player1_win = 0
    player2_win = 0

    # For every epoch
    for i in range(epochs):
        # get the winner
        winner = judge.play(all_states)

        # check which player is the winner
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1

        # print the intermediate win rates, if needed
        if (i + 1) % print_every_n == 0:
            print('Epoch {}: player 1 win rate: {:.2f}, player 2 win rate: {:.2f}'.format(
                i + 1,
                player1_win / print_every_n,
                player2_win / print_every_n
            ))
            player1_win = 0
            player2_win = 0

        # update value estimates of both players
        player1.update_state_value_estimates()
        player2.update_state_value_estimates()

        # reset the judge => players
        judge.reset()

    # Save the players' policies
    player1.save_policy()
    player2.save_policy()

    # endregion Body


def compete(turns):
    # region Summary
    """
    Compete trained RL players
    :param turns: number of turns for competition
    """
    # endregion Summary

    # region Body

    # Create 2 RL players with ε = 0 exploring probability (i.e. greedy)
    player1 = RLPlayer(all_states, epsilon=0)
    player2 = RLPlayer(all_states, epsilon=0)

    # Create a judge to organize the game
    judge = Judge(player1, player2)

    # Load the players' policies
    player1.load_policy()
    player2.load_policy()

    # Set the initial win rate of both players to 0
    player1_win = 0
    player2_win = 0
    tie = 0

    # For every turn
    for _ in range(turns):
        # get the winner
        winner = judge.play(all_states)

        # check which player is the winner
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            tie += 1

        # reset the judge => players
        judge.reset()

    print('Competition result:')
    print('Player 1 win rate: {:.2f}'.format(player1_win / turns))
    print('Player 2 win rate: {:.2f}'.format(player2_win / turns))
    print('Tie rate: {:.2f}'.format(tie / turns))

    # endregion Body


def play():
    # region Summary
    """
    Play against RL player. The game is a 0-sum game. If both players are playing with an optimal strategy, every game will end in a tie.
    So we test whether the RL can guarantee at least a tie if it plays 2nd.
    """
    # endregion Summary

    # region Body

    while True:
        # Create a human player
        human = HumanPlayer()

        # Create RL player with ε = 0 exploring probability (i.e. greedy)
        rl_player = RLPlayer(all_states, epsilon=0)

        # Create a judge to organize the game
        judge = Judge(human, rl_player)

        # Load the RL player's policy
        rl_player.load_policy()

        # Get the winner
        winner = judge.play(all_states, print_state=True)

        # Check which player is the winner
        if winner == 1:
            print('You win!')
        elif winner == -1:
            print('You lose.')
        else:
            print("It's a tie!")

    # endregion Body

# endregion Functions


if __name__ == '__main__':
    #train(epochs=int(1e5))
    #compete(turns=int(1e3))
    play()