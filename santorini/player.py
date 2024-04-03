class Player:
    """Player class to manage player actions and workers."""

    def __init__(self, player_id = "", god_card = None):
        self.player_id = player_id
        self._god_card = god_card

    def __bool__(self):
        return bool(self.player_id)
