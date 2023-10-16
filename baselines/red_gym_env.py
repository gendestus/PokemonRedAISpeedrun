
import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy

from gamestate import Gamestate
from reward_events import RewardEvents

import hnswlib
import mediapy as media
import pandas as pd
import datetime

from gymnasium import Env, spaces, wrappers
from pyboy.utils import WindowEvent

class RedGymEnv(Env):


    def __init__(
        self, config=None):

        self.FEATURES_KEY = "features"
        self.FRAME_BUFFER_KEY = "frame_buffer"
        
        self.previous_map_id = -1
        self.seen_maps = []

        # seen_position_combos are strings containing the map id and the player's position. We award a single point for entering a combo that we haven't seen yet to encourage exploration
        self.seen_position_combos = []

        self.game_start_time = None

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.all_runs = []
        self.last_obs = None

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        #self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)
        self.observation_space = spaces.Dict({self.FEATURES_KEY:spaces.MultiDiscrete([255, 255, 255, 512, 255,255,255,255,255,255,512,255,255,255,255,255,255,255,512,255,255,255,255,255,255,255,512,255,255,255,255,255,255,255,512,255,255,255,255,255,255,255,512,255,255,255,255,255,255,255,512,255,255,255,255,255]), self.FRAME_BUFFER_KEY:spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)})

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )
        self.gamestate = Gamestate(self.pyboy)
        self.reward_events = RewardEvents()

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0 if config['headless'] else 6)
        self.reset()

    # Override from Gym
    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    # Override from Gym
    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []
        
        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
       
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return {self.FEATURES_KEY:np.zeros(56, dtype=int), self.FRAME_BUFFER_KEY:self.render()}, {}
    
    # Override from Gym
    def step(self, action):
        if self.game_start_time == None:
            self.game_start_time = datetime.datetime.now()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        # frame_start = 2 * (self.memory_height + self.mem_padding)
        # obs_flat = obs_memory[
        #     frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        # self.update_frame_knn_index(obs_flat)

        new_reward, new_prog = self.update_reward()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1
        obs = self.get_observables(obs_memory)
        #return obs_memory, new_reward*0.1, False, step_limit_reached, {}
        return obs, new_reward*0.1, False, step_limit_reached, {}
    
    ######################################################
    #   Start non-Override methods
    ######################################################

    def add_video_frame(self):
        '''
        Adds a frame to the video writer. Called every step if save_video is true.
        '''
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def append_agent_stats(self, action):
        '''
        Appends the current agent stats to the agent_stats list. Called every step. 

        NEEDS REWORK FOR SPEEDRUN
        '''
        player_position = self.gamestate.get_player_position()
        map_id = self.gamestate.get_current_map()
        party_pokemon = self.gamestate.get_party()
        levels = []
        for i in range(len(party_pokemon)):
            levels.append(party_pokemon[i]["level"])
        num_pokemon = self.gamestate.get_num_pokemon()
        self.agent_stats.append({
            'step': self.step_count, 'x': player_position[0], 'y': player_position[1], 'map': map_id,
            'last_action': action,
            'pcount': num_pokemon, 'levels': levels, 'badge': self.gamestate.get_num_badges()
        })
    def check_if_done(self):
        '''
        Checks if the episode is done. Called every step.
        '''
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done
    def clamp(self, val, min_val, max_val):
        return min(max(val, min_val), max_val)
    def create_exploration_memory(self):
        '''
        Creates the exploration memory for the current frame. Called every step.
        Probably deprecated for speedrunning since we have map ids
        '''
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.gamestate.get_num_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory
    
    def create_recent_memory(self):
        '''
        Creates the recent memory for the current frame. Called every step.
        '''
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)
    
    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''

        #this needs to be changed to a weighted point value system to guide the AI towards the critical path
        map_id = self.gamestate.get_current_map()
        if map_id not in self.seen_maps:
            self.seen_maps.append(map_id)
        
        player_position = self.gamestate.get_player_position()
        position_combo = str(map_id) + str(player_position)
        if position_combo not in self.seen_position_combos:
            self.seen_position_combos.append(position_combo)

        if self.gamestate.get_current_map() != 40 and not self.reward_events.HasLeftOakLab:
            self.reward_events.HasLeftOakLab = True
            print("LEFT OAK'S LAB FOR THE FIRST TIME")

        state_scores = {
            #'event': self.update_max_event_rew(),  
            #'party_xp': 0.1*sum(poke_xps),
            #'level': self.get_levels_reward(), 
            #'heal': self.total_healing_rew,
            #'op_lvl': self.update_max_op_level(),
            #'dead': -0.1*self.died_count,
            'badge': self.gamestate.get_num_badges() * 2,
            #'op_poke': self.max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': seen_poke_count * 400,
            #'explore': self.get_knn_reward()
            'speedrun_map_progress': self.clamp(len(self.seen_maps) - 1, 0, 200) * 10,
            'speedrun_position_progress': self.clamp(len(self.seen_position_combos) - 1, 0, 200),
            'speedrun_events': self.reward_events.calculate_reward(self.game_start_time)
        }
        
        return state_scores
    
    def get_levels_reward(self):
        '''
        Calculates the reward for the levels of the party pokemon. Called every step.
        Deprecated
        '''
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew
    
    def get_levels_sum(self):
        '''
        Gets the sum of the levels of the party pokemon minus 4 for the starting pokemon. Called every step.
        Possibly deprecated as we move on from level based rewards
        '''
        # poke_levels = [max(self.gamestate.read(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        # return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
        levels = []
        for i in range(6):
            level = int(self.gamestate.get_party_pokemon(i+1)["level"])
            if level > 2:
                levels.append(level - 2)
        return max(sum(levels) - 4, 0)
    
    def get_observables(self, rendered_frame):
        '''
        Gets memory values for passing along to the observation space along with the frame buffer. These represent the input features for the model. Called every step and reset.
        Values are taken from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        '''
        enemy_pokemon_info = self.gamestate.get_enemy_pokemon()
        enemy_pokemon = enemy_pokemon_info["species"]
        enemy_pokemon_type1 = enemy_pokemon_info["type1"]
        enemy_pokemon_type2 = enemy_pokemon_info["type2"]
        enemy_pokemon_health = enemy_pokemon_info["health"]
        map_id = self.gamestate.get_current_map()
        player_position = self.gamestate.get_player_position()
        position_x = player_position[0]
        position_y = player_position[1]
        party = self.gamestate.get_party()
        player_pokemon1 = party[0]["species"]
        player_pokemon1_type1 = party[0]["type1"]
        player_pokemon1_type2 = party[0]["type2"]
        player_pokemon1_health = party[0]["health"]
        player_pokemon1_move1 = party[0]["move1"]
        player_pokemon1_move2 = party[0]["move2"]
        player_pokemon1_move3 = party[0]["move3"]
        player_pokemon1_move4 = party[0]["move4"]
        player_pokemon2 = party[1]["species"]
        player_pokemon2_type1 = party[1]["type1"]
        player_pokemon2_type2 = party[1]["type2"]
        player_pokemon2_health = party[1]["health"]
        player_pokemon2_move1 = party[1]["move1"]
        player_pokemon2_move2 = party[1]["move2"]
        player_pokemon2_move3 = party[1]["move3"]
        player_pokemon2_move4 = party[1]["move4"]
        player_pokemon3 = party[2]["species"]
        player_pokemon3_type1 = party[2]["type1"]
        player_pokemon3_type2 = party[2]["type2"]
        player_pokemon3_health = party[2]["health"]
        player_pokemon3_move1 = party[2]["move1"]
        player_pokemon3_move2 = party[2]["move2"]
        player_pokemon3_move3 = party[2]["move3"]
        player_pokemon3_move4 = party[2]["move4"]
        player_pokemon4 = party[3]["species"]
        player_pokemon4_type1 = party[3]["type1"]
        player_pokemon4_type2 = party[3]["type2"]
        player_pokemon4_health = party[3]["health"]
        player_pokemon4_move1 = party[3]["move1"]
        player_pokemon4_move2 = party[3]["move2"]
        player_pokemon4_move3 = party[3]["move3"]
        player_pokemon4_move4 = party[3]["move4"]
        player_pokemon5 = party[4]["species"]
        player_pokemon5_type1 = party[4]["type1"]
        player_pokemon5_type2 = party[4]["type2"]
        player_pokemon5_health = party[4]["health"]
        player_pokemon5_move1 = party[4]["move1"]
        player_pokemon5_move2 = party[4]["move2"]
        player_pokemon5_move3 = party[4]["move3"]
        player_pokemon5_move4 = party[4]["move4"]
        player_pokemon6 = party[5]["species"]
        player_pokemon6_type1 = party[5]["type1"]
        player_pokemon6_type2 = party[5]["type2"]
        player_pokemon6_health = party[5]["health"]
        player_pokemon6_move1 = party[5]["move1"]
        player_pokemon6_move2 = party[5]["move2"]
        player_pokemon6_move3 = party[5]["move3"]
        player_pokemon6_move4 = party[5]["move4"]
        num_badges = self.gamestate.get_num_badges()

        na = np.array([
            enemy_pokemon,
            enemy_pokemon_health,
            enemy_pokemon_type1, 
            enemy_pokemon_type2, 
            map_id, 
            position_x, 
            position_y, 
            player_pokemon1, 
            player_pokemon1_health, 
            player_pokemon1_move1, 
            player_pokemon1_move2, 
            player_pokemon1_move3, 
            player_pokemon1_move4, 
            player_pokemon1_type1, 
            player_pokemon1_type2,
            player_pokemon2,
            player_pokemon2_health,
            player_pokemon2_move1,
            player_pokemon2_move2,
            player_pokemon2_move3,
            player_pokemon2_move4,
            player_pokemon2_type1,
            player_pokemon2_type2,
            player_pokemon3,
            player_pokemon3_health,
            player_pokemon3_move1,
            player_pokemon3_move2,
            player_pokemon3_move3,
            player_pokemon3_move4,
            player_pokemon3_type1,
            player_pokemon3_type2,
            player_pokemon4,
            player_pokemon4_health,
            player_pokemon4_move1,
            player_pokemon4_move2,
            player_pokemon4_move3,
            player_pokemon4_move4,
            player_pokemon4_type1,
            player_pokemon4_type2,
            player_pokemon5,
            player_pokemon5_health,
            player_pokemon5_move1,
            player_pokemon5_move2,
            player_pokemon5_move3,
            player_pokemon5_move4,
            player_pokemon5_type1,
            player_pokemon5_type2,
            player_pokemon6,
            player_pokemon6_health,
            player_pokemon6_move1,
            player_pokemon6_move2,
            player_pokemon6_move3,
            player_pokemon6_move4,
            player_pokemon6_type1,
            player_pokemon6_type2,
            num_badges])
        if map_id != self.previous_map_id:
            print(f'new map: {map_id}')
            self.previous_map_id = map_id
        self.log_observables(na)
        return {self.FEATURES_KEY:na, self.FRAME_BUFFER_KEY:rendered_frame}
    
    def group_rewards(self):
        '''
        Like I said, still trying to figure out the progress reward thing. This is part of that.
        '''
        prog = self.progress_reward
        # these values are only used by memory
        #return (prog['level'] * 100, self.read_hp_fraction()*2000, prog['explore'] * 160)#(prog['events'], 

        # zeroing this out because I don't think it's relevant but who knows....might have to come back to it
        return (0, 0, 0)
    def log_observables(self, obs: np.array):
        '''
        Logs the observables to a file if there's a change. Ignores player position since that constantly changes. Runs every step
        '''
        def generate_log_string(obs: np.array) -> str:
            labels = {"enemy_pokemon":0,
                  "enemy_pokemon_health":1,
                  "enemy_pokemon_type1":2,
                  "enemy_pokemon_type2":3,
                  "map_id":4,
                  "player_pokemon1":7,
                  "player_pokemon2":15,
                  "player_pokemon3":23,
                  "player_pokemon4":31,
                  "player_pokemon5":39,
                  "player_pokemon6":47,
                  "num_badges":55}
            log_string = ''
            for key, val in labels.items():
                log_string += f'{key}: {obs[val]} |'
            return log_string
        

        log_string = generate_log_string(obs)
        if log_string != self.last_obs:
            self.last_obs = log_string
            with open(self.s_path / Path(f'obs_{self.instance_id}.txt'), 'a') as f:
                f.write(f'{datetime.datetime.now()} :: {log_string}\n')

        pass
        
    def run_action_on_emulator(self, action):
        '''
        Handles input for the emulator. Called every step.
        '''
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    def save_and_print_info(self, done, obs_memory):
        '''
        Saves and prints info. Called every step. If done, also handles saving off final info and closing video writers for rendering. 
        Needs rework to handle database solution for more coherent stat keeping
        '''
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

    def save_screenshot(self, name):
        '''
        Saves a screenshot.
        '''
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
        
    def update_reward(self):
        '''
        Updates the reward values based on game state. Called every step.
        Also keeps track of progress values for some reason. I'm still pulling that thread.
        '''

        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )

    
    
    
    
    

    

    
    
    

    

    

    

    
    
    # def read_m(self, addr):
    #     return self.pyboy.get_memory_value(addr)

    # def read_bit(self, addr, bit: int) -> bool:
    #     # add padding so zero will read '0b100000000' instead of '0b0'
    #     return bin(256 + self.read_m(addr))[-bit-1] == '1'
    # def get_badges(self):
    #     return self.bit_count(self.read_m(0xD356))
    # def read_party(self):
    #     return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
    # def update_heal_reward(self):
    #     cur_health = self.read_hp_fraction()
    #     if cur_health > self.last_health:
    #         if self.last_health > 0:
    #             heal_amount = cur_health - self.last_health
    #             if heal_amount > 0.5:
    #                 print(f'healed: {heal_amount}')
    #                 self.save_screenshot('healing')
    #             self.total_healing_rew += heal_amount * 4
    #         else:
    #             self.died_count += 1
    # def get_all_events_reward(self):
    #     return max(sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)]) - 13, 0)
    # def update_max_op_level(self):
    #     #opponent_level = self.read_m(0xCFE8) - 5 # base level
    #     opponent_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
    #     #if opponent_level >= 7:
    #     #    self.save_screenshot('highlevelop')
    #     self.max_opponent_level = max(self.max_opponent_level, opponent_level)
    #     return self.max_opponent_level * 0.2
    # def update_max_event_rew(self):
    #     cur_rew = self.get_all_events_reward()
    #     self.max_event_rew = max(cur_rew, self.max_event_rew)
    #     return self.max_event_rew
    # def read_hp_fraction(self):
    #     hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
    #     max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
    #     return hp_sum / max_hp_sum
    # def read_hp(self, start):
    #     return 256 * self.read_m(start) + self.read_m(start+1)
    # built-in since python 3.10
    # def bit_count(self, bits):
    #     return bin(bits).count('1')
    # def read_triple(self, start_add):
    #     return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    # def read_bcd(self, num):
    #     return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    # def read_money(self):
    #     return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
    #             100 * self.read_bcd(self.read_m(0xD348)) +
    #             self.read_bcd(self.read_m(0xD349)))
    