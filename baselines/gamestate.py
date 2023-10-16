from pyboy import PyBoy

class Gamestate:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read(self, address):
        return self.pyboy.get_memory_value(address)
    def read_bit(self, address, bit) -> bool:
        return bin(256 + self.read_m(address))[-bit-1] == '1'
    
    def get_badges_bitmask(self):
        '''
        Gets badges as a bitmask
        '''
        return self.read(0xD356)
    
    def get_enemy_pokemon(self) -> dict:
        '''
        Gets enemy pokemon info as a dict
        "species: species id
        "health": current hp
        "type1": type 1
        "type2": type 2
        '''
        species = self.read(0xCFE5)
        health = self.read(0xCFE6) + self.read(0xCFE7)
        type1 = self.read(0xCFEA)
        type2 = self.read(0xCFEB)

        return {"species": species, "health": health, "type1": type1, "type2": type2}
    
    def get_current_map(self):
        '''
        Gets current map id
        '''
        return self.read(0xCFE5)
    
    def get_num_badges(self) -> int:
        '''
        Gets number of badges
        '''
        badges_bitmask = self.get_badges_bitmask()
        return sum([int(x) for x in bin(badges_bitmask)[2:]])
    
    def get_num_pokemon(self) -> int:
        '''
        Gets number of pokemon in party
        '''
        return self.read(0xD163)
    
    def get_party(self) -> list:
        '''
        Gets party pokemon info as a list of dicts
        '''
        party = []
        for i in range(6):
            party.append(self.get_party_pokemon(i+1))
        return party
    
    def get_party_pokemon(self, slot) -> dict:
        '''
        Gets party pokemon info as a dict
        "species": species id
        "health": current hp
        "type1": type 1
        "type2": type 2
        "move1": move id 1
        "move2": move id 2
        "move3": move id 3
        "move4": move id 4
        "level": level
        '''
        species_address = 0x0
        health_address = (0x0, 0x0)
        type1_address = 0x0
        type2_address = 0x0
        move1_address = 0x0
        move2_address = 0x0
        move3_address = 0x0
        move4_address = 0x0
        level_address = 0x0

        match slot:
            case 1:
                species_address = 0xD16B
                health_address = (0xD16C, 0xD16D)
                type1_address = 0xD170
                type2_address = 0xD171
                move1_address = 0xD173
                move2_address = 0xD174
                move3_address = 0xD175
                move4_address = 0xD176
                level_address = 0xD18C
            case 2:
                species_address = 0xD197
                health_address = (0xD198, 0xD199)
                type1_address = 0xD19C
                type2_address = 0xD19D
                move1_address = 0xD19F
                move2_address = 0xD1A0
                move3_address = 0xD1A1
                move4_address = 0xD1A2
                level_address = 0xD1B8
            case 3:
                species_address = 0xD1C3
                health_address = (0xD1C4, 0xD1C5)
                type1_address = 0xD1C8
                type2_address = 0xD1C9
                move1_address = 0xD1CB
                move2_address = 0xD1CC
                move3_address = 0xD1CD
                move4_address = 0xD1CE
                level_address = 0xD1E4
            case 4:
                species_address = 0xD1EF
                health_address = (0xD1F0, 0xD1F1)
                type1_address = 0xD1F4
                type2_address = 0xD1F5
                move1_address = 0xD1F7
                move2_address = 0xD1F8
                move3_address = 0xD1F9
                move4_address = 0xD1FA
                level_address = 0xD210
            case 5:
                species_address = 0xD21B
                health_address = (0xD21C, 0xD21D)
                type1_address = 0xD220
                type2_address = 0xD221
                move1_address = 0xD223
                move2_address = 0xD224
                move3_address = 0xD225
                move4_address = 0xD226
                level_address = 0xD23C
            case 6:
                species_address = 0xD247
                health_address = (0xD248, 0xD249)
                type1_address = 0xD24C
                type2_address = 0xD24D
                move1_address = 0xD24F
                move2_address = 0xD250
                move3_address = 0xD251
                move4_address = 0xD252
                level_address = 0xD268
        
        species = self.read(species_address)
        health = self.read(health_address[0]) + self.read(health_address[1])
        type1 = self.read(type1_address)
        type2 = self.read(type2_address)
        move1 = self.read(move1_address)
        move2 = self.read(move2_address)
        move3 = self.read(move3_address)    
        move4 = self.read(move4_address)
        level = self.read(level_address)

        return {
            "species": species,
            "health": health,
            "type1": type1,
            "type2": type2,
            "move1": move1,
            "move2": move2,
            "move3": move3,
            "move4": move4,
            "level": level}
    
    def get_player_position(self):
        '''
        Gets player position as a tuple
        '''
        x = self.read(0xD363)
        y = self.read(0xD364)
        return (x, y)