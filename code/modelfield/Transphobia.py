import numpy as np

# OK NOW WE HAVE TO CONSTRUCT OUR ATTENTION MODEL.
#
#            _______________
#           |               |
#     ______|    Output     |
#    |      | Probabilities |
#    |      |_______________|
#    |       ___     ___     ___
#    |      |   |   |   |   |   |
#    |______| S |___| L |___| A |
#           |___|   |___|   |___|
#                             | 
#                            _|___
#        ___________        |_FF_|      
#      _|__        |         _|___
#     |_AN_|       |        |_AN_|
#      _|__        |        __|___
#     |_FF_|       |_______|_MHA__|
#      _|__                  _|__
#     |_AN_|                |_AN_|
#     __|___             ______|_____
#    |_MHA__|           |_Masked_MHA_|
#       |                     |
#  _____|_________      ______|_______
# |               |    |              |
# |  Positional   |    | Positional   |
# |  Encoding     |    | Encoding     |
# |_______________|    |______________| 
#      |                       |
#     _|_                     _|_
#    | I |                   | O |
#    |_E_|                   |_E_|
#    |_m_|                   |_m_|
#    |_b_|                   |_b_|
#    |_e_|                   |_e_|
#    |_d_|                   |_d_|
                          
#  Inputs                  Outputs
#                     (shifted right)

#      Figure 1: The Transformer - model architecture.