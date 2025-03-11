"""Santorini command line interface"""
from santorini.game import Game, GameState
from santorini import utils

def run_cli():
    """Runs the command line interface"""
    game = Game()
    # Main loop
    while True:
        # Print the board each time so we can see the current state.
        print(str(game.get_board()))
        state = game.get_state()

        # If game is over, announce winner and stop
        if state == GameState.GAME_OVER:
            if game.get_winner() is not None:
                print(f"Game Over! Winner is Player {game.get_winner().get_id()}")
            break

        # Depending on the current state, ask the user for an action
        if state == GameState.PLAYER_SELECT:
            # Ask user how many players
            print("Currently selecting the number of players.")
            action_str = input("Enter 2 or 3: ")
            try:
                action = int(action_str)
                game.step(action)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

        elif state == GameState.SETUP:
            # Player is placing a worker. Let’s ask for a board position 0..24 (on a 5x5).
            print("Setup phase: place your worker on an empty space (row, col).")
            try:
                p_str = input("Enter the placement as comma separated row, col: ")
                p_row, p_col = tuple(map(int, p_str.split(','))) # split move string
                action = utils.space_position_to_index((p_row, p_col))
                game.step(action)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

        elif state == GameState.PLAYING:
            # The game is in normal play. We expect an action like (worker_id, move_index, build_index).
            print(f"{game.get_current_player()} turn. Input (worker_id, move_index, build_index).")
            try:
                w_id_str = input("Enter worker_id: ")
                worker_id = int(w_id_str)

                m_str = input("Enter the move as comma separated row, col: ")
                m_row, m_col = tuple(map(int, m_str.split(','))) # split move string
                move_index = utils.space_position_to_index((m_row, m_col))

                b_str = input("Enter the build as separated row, col: ")
                b_row, b_col = tuple(map(int, b_str.split(','))) # split build string
                build_index = utils.space_position_to_index((b_row, b_col))

                action = (worker_id, move_index, build_index)
                game.step(action)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

        else:
            # Should not happen unless there's an unhandled state.
            print(f"Unhandled game state: {state}")
            break

if __name__ == "__main__":
    run_cli()
