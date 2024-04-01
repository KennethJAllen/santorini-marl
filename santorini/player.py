class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player = None):
        self.player = player

    def __bool__(self):
        return bool(self.player)
