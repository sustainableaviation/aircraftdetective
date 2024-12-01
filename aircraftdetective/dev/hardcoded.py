# Add K_1 for the Comet 4
    comet4_k1 = 5190 / (np.log(0.94*73480/43410)) # data for the Comet 4 from Aerospaceweb.org
    breguet.loc[breguet['Name'] == 'Comet 4', 'K_1'] = comet4_k1

    # Add K_1 for the Comet 1. Real Flight Data
    comet1_k1 = 2761.4 / (np.log(98370/73000)) # data for the Comet 4 from Aerospaceweb.org
    breguet.loc[breguet['Name'] == 'Comet 1', 'K_1'] = comet1_k1

    comet4_A = breguet.loc[breguet['Name'] == 'Comet 4', 'A']
    comet1_A = breguet.loc[breguet['Name'] == 'Comet 1', 'A']
    breguet.loc[breguet['Name'] == 'Comet 4', 'L/D estimate'] = comet4_A/223.6 # account for lower speed of the Comet 4
    breguet.loc[breguet['Name'] == 'Comet 1', 'L/D estimate'] = comet1_A / 201.4  # account for lower speed of the Comet 1
