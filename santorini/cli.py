"""Santorini command line interface"""
from santorini.game import Game, GameState
from santorini import utils

def main():
    """Runs the command line interface"""
    game = Game()
    # Main loop
    while True:
        # Print the board each time so we can see the current state.
        print(str(game.board))
        state = game.state

        # If game is over, announce winner and stop
        if state == GameState.GAME_OVER:
            if game.winner is not None:
                print(f"Game Over! Winner is Player {game.winner.get_id()}")
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
            print("Setup phase: place your worker on an empty space (x, y).")
            try:
                p_str = input("Enter the placement as comma separated x, y: ")
                p_x, p_y = tuple(map(int, p_str.strip(',').split(','))) # split move string
                action = utils.encode_space((p_x, p_y))
                game.step(action)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

        elif state == GameState.PLAYING:
            # The game is in normal play. We expect an action like (worker_id, move_index, build_index).
            print(f"{game.current_player()} turn. Input (worker_id, move_index, build_index).")
            try:
                worker_id_str = input("Enter worker_id: ")
                worker_id = int(worker_id_str)
                move_from = game.current_player().get_worker(worker_id).position

                move_to_str = input("Enter the move as comma separated x, y: ")
                move_to = tuple(map(int, move_to_str.split(','))) # split move string

                b_str = input("Enter the build as separated x, y: ")
                build_on = tuple(map(int, b_str.split(','))) # split build string

                action = utils.encode_action((move_from, move_to, build_on))
                game.step(action)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

        else:
            # Should not happen unless there's an unhandled state.
            print(f"Unhandled game state: {state}")
            break

if __name__ == "__main__":
    main()
