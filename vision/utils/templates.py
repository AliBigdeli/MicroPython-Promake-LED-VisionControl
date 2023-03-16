# Pre-defined hand gesture templates
class Gesture:
    def __init__(self, label):
        self.gestures = {
            'One (red)':          {'finger states':   [[2], [0], [3, 4], [3, 4], [3, 4]],
                             'direction':       'up',
                             'wrist angle':     [0.65, 0.85],
                             'overlap':         None,
                             'boundary':        None},
            'Two (blue)':   {'finger states':   [[2], [0], [0], [3, 4], [3, 4]],
                             'direction':       'up',
                             'wrist angle':     [0.75, 0.95],
                             'overlap':         None,
                             'boundary':        None},
            'Three (green)':        {'finger states':   [[2], [0], [0], [0], [3, 4]],
                             'direction':       'up',
                             'wrist angle':     [0.70, 0.90],
                             'overlap':         None,
                             'boundary':        None},
            'Four (yellow)':         {'finger states':   [[2], [0], [0], [0], [0]],
                             'direction':       'up',
                             'wrist angle':     [0.70, 0.90],
                             'overlap':         None,
                             'boundary':        None},
            'On (white)':         {'finger states':   [[0], [0], [0], [0], [0]],
                             'direction':       'up',
                             'wrist angle':     [0.70, 0.90],
                             'overlap':         None,
                             'boundary':        None},
            'Off (dark)':          {'finger states':   [[2], [4], [4], [4], [4]],
                             'direction':       'up',
                             'wrist angle':     [0.75, 0.95],
                             'overlap':         None,
                             'boundary':        None},
        }